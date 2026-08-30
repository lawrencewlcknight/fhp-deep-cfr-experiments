"""Ray-parallel traversal collection for Deep CFR.

The implementation mirrors the ESCHER parallelisation pattern used in this
workspace: a single central learner owns optimisation and evaluation, while
Ray actors hold synchronised inference networks and collect external-sampling
traversals in parallel. Traversal budgets are partitioned across workers rather
than multiplied by the worker count.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from .game import load_game
from .parallel_utils import partition_total, worker_seed
from .seeding import set_seed
from .solver import DeepCFRSolver


UPSTREAM_PARALLEL_SOURCE = (
    "https://github.com/Sandholm-Lab/ESCHER/blob/"
    "e694eaaa251952696aaf36ef1c790887c8324750/parallelized_ESCHER.py"
)


class DeepCFRTraversalWorker:
    """Ray actor payload for traversal-only Deep CFR sample generation."""

    def __init__(
        self,
        game_name: str,
        solver_kwargs: Dict[str, Any],
        worker_seed_value: int,
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        set_seed(int(worker_seed_value))
        game = load_game(str(game_name))
        self._solver = DeepCFRSolver(game, **dict(solver_kwargs))

    def ping(self) -> bool:
        return True

    def _load_advantage_weights(self, advantage_state_dicts: List[Dict[str, Any]]) -> None:
        for network, state_dict in zip(
            self._solver._advantage_networks,  # pylint: disable=protected-access
            advantage_state_dicts,
        ):
            network.load_state_dict(state_dict)

    def collect(
        self,
        n: int,
        player: int,
        advantage_state_dicts: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        """Collects traversal samples and returns the worker-local deltas."""
        self._solver.clear_advantage_buffers()
        self._solver.strategy_buffer.clear()
        self._load_advantage_weights(advantage_state_dicts)
        self._solver._iteration = int(iteration)  # pylint: disable=protected-access

        before = self._solver._nodes_touched  # pylint: disable=protected-access
        for _ in range(int(n)):
            self._solver._traverse_game_tree(  # pylint: disable=protected-access
                self._solver._root_node,  # pylint: disable=protected-access
                int(player),
            )

        advantage_memories = [
            list(buffer)
            for buffer in self._solver.advantage_buffers
        ]
        strategy_memories = list(self._solver.strategy_buffer)
        self._solver.clear_advantage_buffers()
        self._solver.strategy_buffer.clear()

        return {
            "nodes_touched": int(
                self._solver._nodes_touched - before  # pylint: disable=protected-access
            ),
            "advantage_memories": advantage_memories,
            "strategy_memories": strategy_memories,
        }


class ParallelDeepCFRSolver(DeepCFRSolver):
    """Single-learner Deep CFR with Ray-parallel traversal collection."""

    def __init__(
        self,
        game,
        *,
        game_name: str,
        parallel_num_workers: int = 3,
        parallel_run_seed: int = 0,
        parallel_ray_address: Optional[str] = None,
        parallel_log_to_driver: bool = False,
        parallel_worker_memory_capacity: Optional[int] = None,
        parallel_ray_object_store_memory: Optional[int] = None,
        **solver_kwargs,
    ) -> None:
        self._parallel_num_workers = int(parallel_num_workers)
        if self._parallel_num_workers < 2:
            raise ValueError("parallel_num_workers must be at least 2.")
        total_memory_capacity = int(solver_kwargs["memory_capacity"])
        if total_memory_capacity < 1:
            raise ValueError("memory_capacity must be positive.")
        if parallel_worker_memory_capacity is None:
            if total_memory_capacity < self._parallel_num_workers:
                raise ValueError(
                    "memory_capacity must provide at least one slot per "
                    "parallel worker."
                )
            worker_capacities = partition_total(
                total_memory_capacity, self._parallel_num_workers
            )
        else:
            worker_capacity = int(parallel_worker_memory_capacity)
            if worker_capacity < 1:
                raise ValueError("parallel_worker_memory_capacity must be positive.")
            worker_capacities = [worker_capacity] * self._parallel_num_workers

        super().__init__(game, **solver_kwargs)
        self._game_name = str(game_name)
        self._parallel_run_seed = int(parallel_run_seed)
        self._workers = []
        self._ray = None
        self._owns_ray_runtime = False

        try:
            import ray

            self._ray = ray
            self._owns_ray_runtime = not ray.is_initialized()
            if self._owns_ray_runtime:
                init_kwargs = {
                    "include_dashboard": False,
                    "log_to_driver": bool(parallel_log_to_driver),
                    "ignore_reinit_error": True,
                }
                if parallel_ray_address:
                    init_kwargs["address"] = str(parallel_ray_address)
                else:
                    init_kwargs["num_cpus"] = self._parallel_num_workers
                    object_store_memory = parallel_ray_object_store_memory
                    if object_store_memory is None:
                        object_store_memory = 512 * 1024 * 1024
                    init_kwargs["object_store_memory"] = int(object_store_memory)
                ray.init(**init_kwargs)

            worker_class = ray.remote(num_cpus=1)(DeepCFRTraversalWorker)
            for worker_index, worker_capacity in enumerate(worker_capacities):
                worker_kwargs = dict(solver_kwargs)
                worker_kwargs.update({
                    "compute_exploitability": False,
                    "memory_capacity": int(worker_capacity),
                })
                self._workers.append(
                    worker_class.remote(
                        self._game_name,
                        worker_kwargs,
                        worker_seed(self._parallel_run_seed, worker_index),
                    )
                )
            ray.get([worker.ping.remote() for worker in self._workers])
        except Exception:
            self.close()
            raise

    @property
    def execution_backend(self) -> str:
        return "ray_parallel"

    @property
    def parallel_num_workers(self) -> int:
        return self._parallel_num_workers

    def close(self) -> None:
        ray = getattr(self, "_ray", None)
        if ray is None:
            return
        for worker in getattr(self, "_workers", []):
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self._workers = []
        if self._owns_ray_runtime and ray.is_initialized():
            ray.shutdown()
        self._ray = None

    def _advantage_state_payload(self) -> List[Dict[str, torch.Tensor]]:
        """Returns CPU-cloned advantage-network weights for Ray workers."""
        return [
            {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            for network in self._advantage_networks
        ]

    def _collect_traversals_for_player(self, player: int) -> None:
        counts = partition_total(self._num_traversals, self._parallel_num_workers)
        weights_ref = self._ray.put(self._advantage_state_payload())
        refs = [
            worker.collect.remote(
                int(count),
                int(player),
                weights_ref,
                int(self._iteration),
            )
            for worker, count in zip(self._workers, counts)
            if int(count) > 0
        ]
        results = self._ray.get(refs) if refs else []

        self._nodes_touched += sum(int(row["nodes_touched"]) for row in results)
        for row in results:
            for player_index, memories in enumerate(row["advantage_memories"]):
                for memory in memories:
                    self._advantage_memories[player_index].add(memory)
            for memory in row["strategy_memories"]:
                self._strategy_memories.add(memory)
        del results, refs, weights_ref

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
