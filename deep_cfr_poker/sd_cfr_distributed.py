"""Synchronous CPU data-parallel fitting for Experiment 5's SD-CFR learner.

The existing actors alternate between traversal and fitting. One central
reservoir samples each GLOBAL minibatch, then targets are standardized before
sharding. Persistent Gloo workers sum gradients before each replicated Adam
step. No local reservoirs, local optimizer steps, or effective-batch increase.
Only floating-point reduction order differs from central fitting.
"""
import copy
from datetime import timedelta
import hashlib
import socket
import time

import numpy as np
import torch
import torch.distributed as dist

from .parallel_utils import partition_total
from .replay import CompactAdvantageReservoirBuffer
from .sd_cfr_parallel import (
    ParallelStructuredSingleDeepCFRSolver, SDCFRTraversalWorker, isolated_rng,
)


def _state_digest(network, optimizer):
    digest = hashlib.sha256()
    for tensor in network.state_dict().values():
        digest.update(tensor.detach().numpy().tobytes())
    for state in optimizer.state_dict()["state"].values():
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.detach().numpy().tobytes() if torch.is_tensor(value)
                          else repr(value).encode())
    return digest.hexdigest()


class SDCFRLearnerActor(SDCFRTraversalWorker):
    """Traversal actor plus persistent synchronous gradient worker (one CPU)."""

    def ping(self):
        return {**super().ping(), "host": socket.gethostname()}

    def initialize_fitting(self, port, world_size, timeout):
        if dist.is_initialized():
            raise RuntimeError("Fitting actor already owns a process group")
        self._fit_store = dist.TCPStore("127.0.0.1", port, is_master=False,
                                       timeout=timedelta(seconds=timeout))
        dist.init_process_group("gloo", store=self._fit_store, rank=self.worker_index,
                                world_size=world_size, timeout=timedelta(seconds=timeout))
        return self.worker_index

    def fit(self, player, iteration, weights, optimizer_state, shard, global_size, capture_gradient=False):
        if not dist.is_initialized() or player not in (0, 1) or iteration < 1:
            raise RuntimeError("Invalid distributed fitting phase")
        started = time.perf_counter()
        self._weight_transport.load(weights)
        network = self.solver._eager_advantages[player]
        optimizer = self.solver._optimizer_advantages[player]
        # Transported tensors must not become read-only/shared Adam storage.
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        network.train()
        parameters = tuple(network.parameters())
        bucket = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float32)
        offset = 0
        for param in parameters:
            param.grad = bucket[offset:offset + param.numel()].view_as(param)
            offset += param.numel()
        n_steps, local_size = shard["iterations"].shape
        if n_steps < 1 or global_size < local_size:
            raise ValueError("Invalid distributed minibatch dimensions")
        # Unpack only this worker's current minibatch, not its whole fit payload.
        codec = self.collectors[player].codec
        last_loss, grad_norms, compute_seconds, communication_seconds = 0.0, [], 0.0, 0.0
        for step in range(n_steps):
            tick = time.perf_counter()
            bucket.zero_()
            if local_size:
                features = torch.from_numpy(codec._unpack(shard["info_states"][step]))
                # Ray object-store arrays are immutable: own the tiny target tensors.
                targets = torch.from_numpy(shard["targets"][step].copy())
                iters = torch.from_numpy(np.sqrt(shard["iterations"][step]).reshape(-1, 1))
                loss = torch.nn.functional.mse_loss(iters * network(features), iters * targets)
                # SUM reduction, with example weighting (also for uneven shards).
                loss = loss * (local_size / global_size)
                loss.backward()
                last_loss = float(loss.detach())
            compute_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
            communication_seconds += time.perf_counter() - tick
            if not torch.isfinite(bucket).all():
                raise FloatingPointError("Non-finite distributed gradient")
            tick = time.perf_counter()
            if self.worker_index == 0:
                grad_norms.append(self.solver._gradient_norm(parameters))
            optimizer.step()
            compute_seconds += time.perf_counter() - tick
        result = dict(worker=self.worker_index, player=player, iteration=iteration,
                      updates=n_steps, examples=n_steps * local_size,
                      last_loss=last_loss, digest=_state_digest(network, optimizer),
                      seconds=time.perf_counter() - started, compute_seconds=compute_seconds,
                      communication_seconds=communication_seconds)
        if self.worker_index == 0:
            result.update(weights=self._weight_transport.snapshot(),
                          optimizer=copy.deepcopy(optimizer.state_dict()),
                          grad_norm=float(np.mean(grad_norms)))
            if capture_gradient:
                result["last_gradient"] = bucket.numpy().copy()
        return result


class DistributedFittingSingleDeepCFRSolver(ParallelStructuredSingleDeepCFRSolver):
    """Experiment 5 unchanged except synchronous fitting on its eight actors.

    Local-Ray only: this is CPU distribution within one VM, not a multi-VM
    cluster. The serial backend remains a central-fitting correctness control.
    Prefetch is safe only for uniform replay and deterministic stateless nets.
    """
    worker_class = SDCFRLearnerActor

    def __init__(self, *args, distributed_fitting=True, **kwargs):
        if distributed_fitting is not True:
            raise ValueError("Use the baseline solver for central fitting")
        if kwargs.get("parallel_ray_address") is not None:
            raise ValueError("Distributed fitting currently requires one local VM")
        self._fitting_store = None
        self._fitting_ready = False
        self.last_distributed_fit = None
        self._capture_fit_gradients = False  # Tests only; no production gradient transport.
        self._last_fit_gradient = None
        self.distributed_fit_totals = dict(phases=0, updates=0, examples=0,
                                          seconds=0.0, preparation_seconds=0.0)
        super().__init__(*args, **kwargs)
        if (self._advantage_replay_sampling != "uniform" or not self._batch_size_advantage
                or self._advantage_network_train_steps < 1
                or self._advantage_network_type != "residual_layer_norm_centered_advantage_mlp"):
            raise ValueError("Distributed fitting requires the deterministic Experiment 5 recipe")
        self.archive.metadata["parallel_execution"]["fitting"] = dict(
            backend="gloo_synchronous_allreduce" if self._parallel_backend == "ray" else "central_reference",
            workers=self.parallel_num_workers, global_batch_size=self._batch_size_advantage,
            updates_per_player=self._advantage_network_train_steps,
            replay_sampling="central_uniform_without_replacement", target_normalization="global_minibatch",
            gradient_reduction="sum_example_weighted", optimizer="replicated_synchronous_adam",
            prefetched_packed_batches=True, actor_local_reservoirs=False)

    def _start_fitting(self):
        self._start_workers()
        if self._fitting_ready:
            return
        try:
            with isolated_rng():
                if not dist.is_available() or not dist.is_gloo_available():
                    raise RuntimeError("This PyTorch build does not support Gloo CPU fitting")
                workers = self._ray.get([w.ping.remote() for w in self._workers],
                                        timeout=self._parallel_timeout)
                if any(w["host"] != socket.gethostname() for w in workers):
                    raise RuntimeError("Distributed fitting actors must share the learner VM")
                self._fitting_store = dist.TCPStore("127.0.0.1", 0, is_master=True,
                    wait_for_workers=False, timeout=timedelta(seconds=self._parallel_timeout))
                ranks = self._ray.get([w.initialize_fitting.remote(
                    self._fitting_store.port, self.parallel_num_workers, self._parallel_timeout)
                    for w in self._workers], timeout=self._parallel_timeout)
                if ranks != list(range(self.parallel_num_workers)):
                    raise RuntimeError("Incorrect distributed fitting ranks")
            self._fitting_ready = True
        except Exception:
            self.close()
            raise

    def _prepare_fit(self, player):
        """Sample the exact baseline batches/RNG stream; keep features packed."""
        buffer = self.advantage_buffers[player]
        n = min(self._batch_size_advantage, len(buffer))
        counts = partition_total(n, self.parallel_num_workers)
        steps = self._advantage_network_train_steps
        shards = [dict(info_states=np.empty((steps, count, buffer._info_state_size), np.uint8),
                       targets=np.empty((steps, count, self._num_actions), np.float32),
                       iterations=np.empty((steps, count), np.float32)) for count in counts]
        self._last_advantage_priority_effective_sample_size[player] = float(len(buffer))
        for step in range(steps):
            # Parent methods bypass dense decode, retaining the baseline's
            # full-buffer fallback and random.sample consumption exactly.
            batch = (CompactAdvantageReservoirBuffer.as_batch(buffer)
                     if self._batch_size_advantage > len(buffer)
                     else CompactAdvantageReservoirBuffer.sample_batch(buffer, n))
            targets = self._process_advantage_targets(batch["targets"], player)
            start = 0
            for count, shard in zip(counts, shards):
                stop = start + count
                shard["info_states"][step] = batch["info_states"][start:stop]
                shard["targets"][step] = targets[start:stop]
                shard["iterations"][step] = batch["iterations"][start:stop]
                start = stop
        return shards, n

    def _learn_advantage_network(self, player):
        if self._parallel_backend == "serial":
            return super()._learn_advantage_network(player)
        if not len(self.advantage_buffers[player]):
            self._last_advantage_priority_effective_sample_size[player] = float("nan")
            return None
        started = time.perf_counter()
        self._start_fitting()
        tick = time.perf_counter()
        shards, global_size = self._prepare_fit(player)
        preparation_seconds = time.perf_counter() - tick
        try:
            with isolated_rng():
                # One actor RPC per complete fit, not one RPC per optimizer step.
                weights = self._ray.put(self._weight_transport.snapshot())
                optimizer = self._ray.put(self._optimizer_advantages[player].state_dict())
                replies = self._ray.get([w.fit.remote(player, self._iteration, weights, optimizer,
                    shard, global_size, self._capture_fit_gradients) for w, shard in zip(self._workers, shards)],
                    timeout=self._parallel_timeout)
            for index, reply in enumerate(replies):
                if (reply["worker"] != index or reply["player"] != player
                        or reply["iteration"] != self._iteration
                        or reply["updates"] != self._advantage_network_train_steps
                        or reply["digest"] != replies[0]["digest"]
                        or not np.isfinite(reply["last_loss"])):
                    raise RuntimeError("Distributed fitting phase/replica validation failed")
            examples = self._advantage_network_train_steps * global_size
            if sum(r["examples"] for r in replies) != examples:
                raise RuntimeError("Distributed fitting lost or duplicated examples")
            # Do not mutate the learner until ALL replicas have succeeded.
            self._weight_transport.load(replies[0]["weights"])
            self._optimizer_advantages[player].load_state_dict(copy.deepcopy(replies[0]["optimizer"]))
            self._advantage_networks[player].train()
            self._last_advantage_grad_norm[player] = replies[0]["grad_norm"]
            self._last_fit_gradient = replies[0].get("last_gradient")
            self.last_distributed_fit = dict(player=player, iteration=self._iteration,
                updates=self._advantage_network_train_steps, examples=examples,
                seconds=time.perf_counter() - started, preparation_seconds=preparation_seconds,
                worker_compute_seconds=[r["compute_seconds"] for r in replies],
                worker_communication_seconds=[r["communication_seconds"] for r in replies])
            self.distributed_fit_totals["phases"] += 1
            for key in ("updates", "examples", "seconds", "preparation_seconds"):
                self.distributed_fit_totals[key] += self.last_distributed_fit[key]
            return sum(r["last_loss"] for r in replies)
        except Exception:
            self.close()
            raise

    def close(self):
        # Kill all actors first: a failed rank must not leave peers waiting in
        # collective teardown. Actor process exit releases its Gloo group.
        super().close()
        self._fitting_ready = False
        self._fitting_store = None
