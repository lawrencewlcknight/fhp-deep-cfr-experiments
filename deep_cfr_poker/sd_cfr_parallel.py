"""Synchronous Ray traversal collection for raw and structured FHP SD-CFR.

Based on parallel_solver's central-learner/worker design, but workers emit
every observation (no local reservoir). Only the learner samples replay and
trains/archives networks. The serial backend is a same-stream correctness
reference, not the historical sequential solver's random trajectory.
"""
from contextlib import contextmanager
from collections import OrderedDict
import math
import random
import time

import numpy as np
import torch

from .game import load_fhp_game
from .parallel_utils import partition_total
from .sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from .sd_cfr_structured import StructuredSingleDeepCFRSolver


@contextmanager
def isolated_rng(seed=None):
    """Keep worker/runtime randomness out of the central replay/optimiser RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def phase_seed(run_seed, iteration, player, worker):
    """Independent reproducible streams, unaffected by completion order."""
    values = [int(run_seed), int(iteration), int(player), int(worker)]
    if min(values) < 0:
        raise ValueError("Seed components must be non-negative")
    return int(np.random.SeedSequence(values).generate_state(1)[0])


class _FlatWeights:
    """Immutable contiguous snapshots; worker views share live network storage.

    No matrix arithmetic or dtype conversion: copies preserve all weight bits.
    Views are safe only for the independent, non-reinitialised CPU networks
    supported by the parallel solver. Workers never perform backward passes.
    """
    def __init__(self, networks):
        states = [network.state_dict() for network in networks]
        self.layout = tuple(tuple((key, tuple(value.shape)) for key, value in state.items())
                            for state in states)
        self.views = [[value.detach().numpy() for value in state.values()] for state in states]
        if any(value.dtype != np.float32 for values in self.views for value in values):
            raise ValueError("Parallel weight transport requires float32 CPU networks")
        self.sizes = [sum(value.size for value in values) for values in self.views]

    def snapshot(self):
        arrays = [np.concatenate([value.reshape(-1) for value in values]) for values in self.views]
        for array in arrays:
            array.flags.writeable = False
        return {"format": "flat_float32_v1", "layout": self.layout, "arrays": arrays}

    def load(self, snapshot):
        if snapshot.get("format") != "flat_float32_v1" or snapshot.get("layout") != self.layout:
            raise ValueError("Parallel weight snapshot layout mismatch")
        arrays = snapshot["arrays"]
        if len(arrays) != len(self.views) or any(
            value.dtype != np.float32 or value.shape != (size,)
            for value, size in zip(arrays, self.sizes)
        ):
            raise ValueError("Parallel weight snapshot shape/dtype mismatch")
        for values, array in zip(self.views, arrays):
            offset = 0
            for value in values:
                np.copyto(value, array[offset:offset + value.size].reshape(value.shape), casting="no")
                offset += value.size


class _PhaseInferenceCache:
    """Memoise only identical deterministic inference within frozen phases."""
    def __init__(self, *args, inference_cache_entries=4096, **kwargs):
        self._inference_cache_entries = int(inference_cache_entries)
        self._phase_cache = None
        self.inference_cache_hits = self.inference_cache_misses = 0
        super().__init__(*args, **kwargs)

    def _collect_traversals_for_player(self, player):
        self.inference_cache_hits = self.inference_cache_misses = 0
        self._phase_cache = OrderedDict() if self._inference_cache_entries else None
        try:
            return super()._collect_traversals_for_player(player)
        finally:
            # Both players' weights can change before the next phase.
            self._phase_cache = None

    def _sample_action_from_advantage(self, state, player, *, info_state=None):
        if self._phase_cache is None:
            return super()._sample_action_from_advantage(state, player, info_state=info_state)
        if info_state is None:
            info_state = self._information_state(state, player)
        info_state = np.asarray(info_state, dtype=np.float32)
        key = (player, tuple(state.legal_actions(player)), info_state.tobytes())
        cached = self._phase_cache.get(key)
        if cached is not None:
            self._phase_cache.move_to_end(key)
            self.inference_cache_hits += 1
            return cached
        value = super()._sample_action_from_advantage(state, player, info_state=info_state)
        value[1].flags.writeable = False
        self._phase_cache[key] = value
        if len(self._phase_cache) > self._inference_cache_entries:
            self._phase_cache.popitem(last=False)
        self.inference_cache_misses += 1
        return value


class _RawTraversalSolver(_PhaseInferenceCache, OptimisedSingleDeepCFRSolver):
    pass


class _StructuredTraversalSolver(_PhaseInferenceCache, StructuredSingleDeepCFRSolver):
    pass


class _TraversalRows:
    """Append-only bounded chunks, never a second-stage reservoir sampler."""
    def __init__(self, codec, *, chunk_rows, max_rows, workspace=None):
        chunk_rows = min(chunk_rows, max_rows)
        self.codec, self.chunk_rows, self.max_rows = codec, chunk_rows, max_rows
        if workspace is None:
            workspace = (np.empty((chunk_rows, codec.feature_count), dtype=np.float32),
                         np.empty(chunk_rows, dtype=np.int32),
                         np.empty((chunk_rows, codec._target_size), dtype=np.float32))
        self.features, self.iterations, self.targets = workspace
        self.clear()

    def clear(self):
        self.count, self.used, self.chunks = 0, 0, []

    def __len__(self):
        return self.count

    def add(self, row):
        if self.count >= self.max_rows:
            raise RuntimeError("Worker row limit exceeded; refusing to discard traversal samples")
        self.features[self.used] = row.info_state
        self.iterations[self.used] = row.iteration
        self.targets[self.used] = row.advantage
        self.used += 1
        self.count += 1
        if self.used == self.chunk_rows:
            self.flush()

    def flush(self):
        if self.used:
            self.chunks.append(dict(info_states=self.codec._pack(self.features[:self.used]),
                                    iterations=self.iterations[:self.used].copy(),
                                    targets=self.targets[:self.used].copy()))
            self.used = 0


class SDCFRTraversalWorker:
    """Small inference replicas; no full-size replay, training or archive growth."""
    def __init__(self, *, structured, solver_kwargs, worker_index,
                 run_seed, chunk_rows=4096, max_rows=1_000_000,
                 configure_actor_threads=False, inference_cache_entries=4096):
        # Ray actors are separate processes. The serial reference must not
        # change its caller's global Torch configuration.
        if configure_actor_threads:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        self.worker_index, self.run_seed = int(worker_index), int(run_seed)
        if chunk_rows < 1 or max_rows < 1:
            raise ValueError("Worker row limits must be positive")
        cls = _StructuredTraversalSolver if structured else _RawTraversalSolver
        # Initialization is overwritten by the learner before every phase.
        # Do not allocate each actor's production replay or perturb driver RNG.
        with isolated_rng(phase_seed(run_seed, 0, 0, worker_index)):
            self.solver = cls(inference_cache_entries=inference_cache_entries,
                              **{**solver_kwargs, "memory_capacity": 1})
        self._weight_transport = _FlatWeights(self.solver._eager_advantages)
        # Only one traverser's collector is active in a phase. Share its dense
        # scratch workspace; flushed payloads already own independent copies.
        self.collectors, workspace = [], None
        for codec in self.solver.advantage_buffers:
            collector = _TraversalRows(codec, chunk_rows=int(chunk_rows), max_rows=int(max_rows),
                                       workspace=workspace)
            self.collectors.append(collector)
            workspace = (collector.features, collector.iterations, collector.targets)
        self.solver._advantage_memories = self.collectors

    def ping(self):
        return {"worker": self.worker_index, "torch_threads": torch.get_num_threads()}

    def collect(self, n, player, weights, iteration):
        if int(n) != n or n < 1 or player not in (0, 1) or iteration < 1:
            raise ValueError("Invalid collection phase")
        for collector in self.collectors:
            collector.clear()
        if isinstance(weights, dict):
            self._weight_transport.load(weights)
        else:
            # Keep the previous transport available for equivalence tests.
            for network, state in zip(self.solver._eager_advantages, weights, strict=True):
                network.load_state_dict(state, strict=True)
        solver = self.solver
        solver._iteration, solver._num_traversals, solver._nodes_touched = int(iteration), int(n), 0
        started = time.perf_counter()
        try:
            with isolated_rng(phase_seed(self.run_seed, iteration, player, self.worker_index)):
                solver._collect_traversals_for_player(player)
            collector = self.collectors[player]
            collector.flush()
            if self.collectors[1 - player].count or solver.strategy_buffer.add_calls:
                raise RuntimeError("SD-CFR workers must collect traverser advantages only")
            return dict(worker=self.worker_index, player=player, iteration=iteration,
                        traversals=int(n), nodes_touched=solver._nodes_touched,
                        rows=collector.count, chunks=list(collector.chunks),
                        feature_encoding=collector.codec.feature_encoding,
                        inference_cache_hits=solver.inference_cache_hits,
                        inference_cache_misses=solver.inference_cache_misses,
                        collection_seconds=time.perf_counter() - started)
        finally:
            # Returned chunks own their arrays. Never retain completed phase data.
            for collector in self.collectors:
                collector.clear()


class _ParallelSDCFR:
    _structured_workers = False

    def __init__(self, game=None, *, parallel_num_workers=3, parallel_run_seed=0,
                 parallel_backend="ray", parallel_chunk_rows=4096,
                 parallel_max_rows_per_worker=1_000_000,
                 parallel_timeout_seconds=300, parallel_ray_address=None,
                 parallel_ray_object_store_memory=512 * 1024 * 1024,
                 parallel_inference_cache_entries=None, **kwargs):
        # Suit-canonical inputs repeat more often; raw-card cache hit rates
        # were too low to justify enabling the lookup overhead by default.
        if parallel_inference_cache_entries is None:
            parallel_inference_cache_entries = 4096 if self._structured_workers else 0
        if parallel_backend not in {"ray", "serial"}:
            raise ValueError("parallel_backend must be ray or serial")
        for name, value, minimum in (("workers", parallel_num_workers, 1),
                                    ("run seed", parallel_run_seed, 0),
                                    ("chunk rows", parallel_chunk_rows, 1),
                                    ("inference cache entries", parallel_inference_cache_entries, 0),
                                    ("worker row limit", parallel_max_rows_per_worker, 1)):
            if int(value) != value or value < minimum:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(parallel_timeout_seconds) or parallel_timeout_seconds <= 0:
            raise ValueError("parallel_timeout_seconds must be positive and finite")
        if parallel_ray_object_store_memory < 80 * 1024 * 1024:
            raise ValueError("Ray object store must have at least 80 MiB")
        if game is not None and str(game) != str(load_fhp_game()):
            raise ValueError("Parallel SD-CFR currently supports the frozen FHP game only")
        if kwargs.get("pack_replay", True) is not True:
            raise ValueError("Parallel SD-CFR requires lossless packed replay")
        self._workers, self._ray, self._owns_ray_runtime = [], None, False
        self._closed = False
        self._parallel_num_workers = int(parallel_num_workers)
        self._parallel_backend, self._parallel_run_seed = parallel_backend, int(parallel_run_seed)
        self._parallel_timeout = float(parallel_timeout_seconds)
        self._ray_address = parallel_ray_address
        self._object_store_memory = int(parallel_ray_object_store_memory)
        super().__init__(game, **kwargs)
        self._weight_transport = _FlatWeights(self._eager_advantages)
        # The archive records the resolved learner recipe, not the actor's tiny
        # allocation. No production hyperparameter is replaced on the driver.
        self._worker_kwargs = dict(structured=self._structured_workers,
                                   solver_kwargs=dict(self.archive.metadata["solver_config"]),
                                   run_seed=self._parallel_run_seed,
                                   chunk_rows=int(parallel_chunk_rows),
                                   inference_cache_entries=int(parallel_inference_cache_entries),
                                   max_rows=int(parallel_max_rows_per_worker))
        self.archive.metadata["parallel_execution"] = dict(
            backend=self.execution_backend, workers=self.parallel_num_workers,
            run_seed=self._parallel_run_seed, seed_scheme="seedsequence_run_iteration_player_worker_v1",
            merge_order="worker_index_then_traversal_order", worker_reservoir_sampling=False,
            weight_transport="flat_float32_v1", inference_cache_entries=int(parallel_inference_cache_entries),
            chunk_rows=int(parallel_chunk_rows), max_rows_per_worker=int(parallel_max_rows_per_worker))
        self.last_parallel_collection = None

    @property
    def execution_backend(self):
        return "ray_parallel_sd_cfr" if self._parallel_backend == "ray" else "serial_worker_reference_sd_cfr"

    @property
    def parallel_num_workers(self):
        return self._parallel_num_workers

    def _start_workers(self):
        if self._closed:
            raise RuntimeError("Parallel solver is closed; construct a fresh solver")
        if self._workers:
            return
        try:
            with isolated_rng():
                if self._parallel_backend == "serial":
                    self._workers = [SDCFRTraversalWorker(worker_index=i, **self._worker_kwargs)
                                     for i in range(self.parallel_num_workers)]
                    return
                import ray
                self._ray = ray
                self._owns_ray_runtime = not ray.is_initialized() and self._ray_address is None
                if not ray.is_initialized():
                    options = dict(include_dashboard=False, log_to_driver=False)
                    if self._ray_address:
                        options["address"] = self._ray_address
                    else:
                        options.update(num_cpus=self.parallel_num_workers,
                                       object_store_memory=self._object_store_memory)
                    ray.init(**options)
                actor = ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0,
                                   runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                                            "OPENBLAS_NUM_THREADS": "1"}})(SDCFRTraversalWorker)
                for index in range(self.parallel_num_workers):
                    self._workers.append(actor.remote(worker_index=index, configure_actor_threads=True,
                                                      **self._worker_kwargs))
                ray.get([worker.ping.remote() for worker in self._workers], timeout=self._parallel_timeout)
        except Exception:
            self.close()
            raise

    def _validate_result(self, row, *, worker, count, player):
        codec = self.advantage_buffers[player]
        if any(row[key] != expected for key, expected in
               (("worker", worker), ("player", player), ("iteration", self._iteration),
                ("traversals", count), ("feature_encoding", codec.feature_encoding))):
            raise RuntimeError("Stale or mismatched parallel SD-CFR collection phase")
        if row["nodes_touched"] < count:
            raise RuntimeError("Invalid worker node count")
        size = 0
        for batch in row["chunks"]:
            x, iterations, targets = (batch[k] for k in ("info_states", "iterations", "targets"))
            n = len(iterations)
            if (x.dtype != np.uint8 or x.shape != (n, codec._info_state_size)
                    or iterations.dtype != np.int32 or iterations.shape != (n,)
                    or not np.all(iterations == self._iteration)
                    or targets.dtype != np.float32 or targets.shape != (n, self._num_actions)
                    or not np.isfinite(targets).all()):
                raise RuntimeError("Invalid parallel SD-CFR replay payload")
            size += n
        if size != row["rows"] or size > self._worker_kwargs["max_rows"]:
            raise RuntimeError("Parallel traversal samples lost or row limit exceeded")

    def _collect_traversals_for_player(self, player):
        self._start_workers()
        started = time.perf_counter()
        counts = partition_total(self._num_traversals, self.parallel_num_workers)
        active = [(index, count) for index, count in enumerate(counts) if count]
        weights = self._weight_transport.snapshot()
        # Do not advance learner/replay RNG when invoking runtime or actors.
        with isolated_rng():
            if self._parallel_backend == "serial":
                results = [self._workers[index].collect(count, player, weights, self._iteration)
                           for index, count in active]
            else:
                reference = self._ray.put(weights)
                refs = [self._workers[index].collect.remote(count, player, reference, self._iteration)
                        for index, count in active]
                results = self._ray.get(refs, timeout=self._parallel_timeout)
        # Validate every response before central replay mutation. ray.get(list)
        # preserves submission order, not completion order.
        for (index, count), row in zip(active, results, strict=True):
            self._validate_result(row, worker=index, count=count, player=player)
        for row in results:
            for batch in row["chunks"]:
                self.advantage_buffers[player].add_packed_batch(batch, feature_encoding=row["feature_encoding"])
        nodes = sum(row["nodes_touched"] for row in results)
        self._nodes_touched += nodes
        self.last_parallel_collection = dict(player=player, iteration=self._iteration,
            traversals=sum(counts), worker_traversals=counts, nodes_touched=nodes,
            rows=sum(row["rows"] for row in results), seconds=time.perf_counter() - started,
            inference_cache_hits=sum(row["inference_cache_hits"] for row in results),
            inference_cache_misses=sum(row["inference_cache_misses"] for row in results),
            worker_seconds=[row["collection_seconds"] for row in results])

    def solve(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Parallel solver is closed; construct a fresh solver")
        try:
            return super().solve(*args, **kwargs)
        except Exception:
            # A partially completed iteration must not be silently retried.
            self.close()
            raise

    def close(self):
        self._closed = True
        if self._ray is not None:
            for worker in self._workers:
                try:
                    self._ray.kill(worker, no_restart=True)
                except Exception:
                    pass
            if self._owns_ray_runtime and self._ray.is_initialized():
                self._ray.shutdown()
        self._workers = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ParallelSingleDeepCFRSolver(_ParallelSDCFR, OptimisedSingleDeepCFRSolver):
    """Raw 190-value FHP inputs, one learner and parallel traversal actors."""


class ParallelStructuredSingleDeepCFRSolver(_ParallelSDCFR, StructuredSingleDeepCFRSolver):
    """Experiment 3/4's exact structured encoder and packed replay, in parallel."""
    _structured_workers = True
