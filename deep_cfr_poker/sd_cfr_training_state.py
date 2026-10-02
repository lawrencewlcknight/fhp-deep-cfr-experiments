"""Final-only, lossless continuation state for the parallel structured SD-CFR.

Policy archives are not training states. This format additionally retains Adam,
packed replay (including Algorithm R counters), all central RNGs and learner
counters. Ray actors are reconstructed: their streams are deterministically
derived from run seed / global iteration / player / worker at each phase.
Only load trusted checkpoints produced by this code (learner.pt uses pickle).
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path
import shutil

import numpy as np
import torch

from .sd_cfr_disk import DiskArchiveReader, DiskSDCFRArchive, sha256, write_json
from .sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from .solver import DeepCFRSolver

FORMAT = "fhp_parallel_sd_cfr_training_state_v1"
EXTRA_ATTRIBUTES = (
    "_cumulative_traversal_collection_seconds", "_last_advantage_grad_norm",
    "_last_policy_grad_norm", "_warned_advantage_buffer_too_small",
    "_warned_strategy_buffer_too_small", "_nodes_touched_history",
    "_average_policy_value_history", "last_parallel_collection",
)


def _canonical(value):
    return json.loads(json.dumps(value))


def implementation_hash():
    root = Path(__file__).parent
    names = ("solver.py", "single_solver.py", "sd_cfr.py", "sd_cfr_optimised.py",
             "sd_cfr_structured.py", "sd_cfr_parallel.py", "sd_cfr_disk.py",
             "sd_cfr_training_state.py", "replay.py", "networks.py", "fhp_features.py",
             "parallel_utils.py", "game.py", "seeding.py")
    return hashlib.sha256(json.dumps({name: sha256(root / name) for name in names},
                                    sort_keys=True).encode()).hexdigest()


def runtime_contract():
    from importlib.metadata import version
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, open_spiel=version("open_spiel"), ray=version("ray"),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                device="cpu")


def _file_record(path):
    return dict(path=path.name, size_bytes=path.stat().st_size, sha256=sha256(path))


def _contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Training-state path escapes its worker directory")
    return path


def _verify_file(root, record):
    path = _contained(root, record["path"])
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Training-state integrity mismatch: {path}")
    return path


def save_training_state(solver, worker_root, checkpoint_path, *, config, execution, context):
    """Write one final state, reusing the adjacent immutable strategy archive.

    Called after solve() returns, including its final diagnostic/RNG work.
    The ready manifest is committed last; an interrupted write is never resumable.
    Replay arrays are streamed directly in packed form, without dense expansion
    or constructing a second multi-million-row replay in memory.
    """
    root = Path(worker_root).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_relative_to(root) or checkpoint.parent != root / "archive":
        raise ValueError("Expected an adjacent policy archive in this worker")
    if not isinstance(solver.archive, DiskSDCFRArchive):
        raise ValueError("A disk-backed historical strategy archive is required")
    if solver._iteration != solver.archive.count + 1 or solver.archive.partial:
        raise ValueError("Save only after a complete player-pair iteration")
    if solver.strategy_buffer.add_calls or len(solver.strategy_buffer):
        raise ValueError("Standalone SD-CFR must not collect policy replay")
    if any(next(n.parameters()).device.type != "cpu" for n in solver._advantage_networks):
        raise ValueError("This training-state format supports the CPU experiment only")
    reader = DiskArchiveReader(checkpoint, solver._game)
    if reader.count != solver.archive.count:
        raise ValueError("Policy checkpoint is not the learner's final archive prefix")
    destination = root / "training_state"
    destination.mkdir(exist_ok=False)
    payload = DeepCFRSolver.extract_full_model(solver, include_buffers=False, include_rng_state=True)
    payload.update(extra_attributes={key: copy.deepcopy(getattr(solver, key)) for key in EXTRA_ATTRIBUTES},
                   advantage_training_modes=[net.training for net in solver._advantage_networks],
                   policy_training_mode=solver._policy_network.training,
                   runner_context=copy.deepcopy(context))
    path = destination / "learner.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    buffers = []
    for player, buffer in enumerate(solver.advantage_buffers):
        meta = dict(player=player, capacity=buffer.capacity, size=len(buffer),
                    add_calls=buffer.add_calls, feature_encoding=buffer.feature_encoding,
                    feature_count=buffer.feature_count, arrays={})
        for field, array in (("info_states", buffer._info_states), ("iterations", buffer._iterations),
                             ("targets", buffer._targets)):
            path = destination / f"replay_p{player}_{field}.npy"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array[:len(buffer)], allow_pickle=False)
            temporary.replace(path)
            meta["arrays"][field] = _file_record(path)
        buffers.append(meta)
    manifest = dict(format=FORMAT, status="complete", config=_canonical(config),
                    execution=_canonical(execution), runtime=runtime_contract(),
                    implementation_sha256=implementation_hash(),
                    seed=context["seed"], smoke=context["smoke"], hours=context["hours"],
                    active_seconds=context["active_seconds"],
                    next_iteration=solver._iteration, nodes_touched=solver._nodes_touched,
                    archive_checkpoint=str(checkpoint.relative_to(root)),
                    archive_checkpoint_sha256=sha256(checkpoint),
                    archive_chunk_iterations=solver.archive.chunk_iterations,
                    learner=_file_record(destination / "learner.pt"), buffers=buffers,
                    archive_storage="adjacent_immutable_chunks_stored_once",
                    actor_state="reconstruct_from_phase_seed_and_next_global_iteration")
    path = destination / "manifest.json"
    write_json(path, manifest)
    return path


def inspect_training_state(path, *, expected_config=None, expected_execution=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Not a complete resumable SD-CFR training state")
    if path.parent.name != "training_state":
        raise ValueError("Keep training_state/ and archive/ adjacent within the worker")
    if manifest["implementation_sha256"] != implementation_hash():
        raise ValueError("Training implementation differs; use the same saved code revision")
    if manifest["runtime"] != runtime_contract():
        raise ValueError("Training runtime differs; use the saved Python/library versions and threads")
    for name, expected in (("config", expected_config), ("execution", expected_execution)):
        if expected is not None and manifest[name] != _canonical(expected):
            raise ValueError(f"Resume {name} differs from the saved training contract")
    return manifest


def load_training_state(path, output_archive, *, expected_config=None, expected_execution=None,
                        solver_class=ParallelStructuredSingleDeepCFRSolver):
    """Restore into a NEW archive directory, leaving the original run untouched.

    Returns (solver, runner_context). Call solver.close() when finished. Replay
    arrays are checksum-verified and mmap-read before copying into the allocated
    buffers. The archive is copied once, then extended by new immutable chunks.
    """
    path = Path(path).resolve()
    source_root = path.parent.parent
    meta = inspect_training_state(path, expected_config=expected_config,
                                  expected_execution=expected_execution)
    learner = _verify_file(path.parent, meta["learner"])
    checked_arrays = []
    if len(meta["buffers"]) != 2:
        raise ValueError("Two complete advantage reservoirs are required")
    for player, entry in enumerate(meta["buffers"]):
        if entry["player"] != player or not 0 <= entry["size"] <= entry["capacity"] <= 5_000_000:
            raise ValueError("Invalid replay player/size/capacity")
        if entry["add_calls"] < entry["size"]:
            raise ValueError("Invalid reservoir stream counter")
        checked_arrays.append({field: _verify_file(path.parent, record)
                               for field, record in entry["arrays"].items()})
    checkpoint = _contained(source_root, meta["archive_checkpoint"])
    if checkpoint.parent != source_root / "archive" or sha256(checkpoint) != meta["archive_checkpoint_sha256"]:
        raise ValueError("Final policy manifest integrity mismatch")
    target = Path(output_archive).resolve()
    if target.is_relative_to(source_root) or source_root.is_relative_to(target):
        raise ValueError("Resume output must be separate from the source worker")
    payload = torch.load(learner, map_location="cpu", weights_only=False)
    context = payload["runner_context"]
    if (payload["iteration"] != meta["next_iteration"]
            or payload["training_state"]["nodes_touched"] != meta["nodes_touched"]
            or any(context[key] != meta[key] for key in ("seed", "smoke", "hours", "active_seconds"))):
        raise ValueError("Learner and manifest counters/context differ")
    solver = solver_class(pack_replay=True, **meta["config"], **meta["execution"])
    try:
        reader = DiskArchiveReader(checkpoint, solver._game)
        if reader.count + 1 != meta["next_iteration"]:
            raise ValueError("Historical strategy archive is not aligned with the learner")
        archive = DiskSDCFRArchive(solver, target, chunk_iterations=meta["archive_chunk_iterations"])
        if _canonical(archive.contract) != {k: v for k, v in reader.contract.items()
                                           if k not in ("completed_iterations", "chunks")}:
            raise ValueError("Historical archive metadata differs from the restored learner")
        for chunk in reader.chunks:
            shutil.copy2(checkpoint.parent / chunk["path"], target / chunk["path"])
        # Retain every prior playable prefix, without copying archive chunks twice.
        for row in context["records"]:
            old = _contained(source_root, row["path"])
            if old.parent != checkpoint.parent or sha256(old) != row["sha256"]:
                raise ValueError("Prior playable checkpoint integrity mismatch")
            shutil.copy2(old, target / old.name)
        archive.chunks = copy.deepcopy(reader.chunks)
        archive.count = reader.count
        solver.archive = archive
        DeepCFRSolver.load_full_model(solver, payload, restore_buffers=False, restore_rng_state=False)
        for buffer, entry, files in zip(solver.advantage_buffers, meta["buffers"], checked_arrays):
            if (buffer.capacity != entry["capacity"] or buffer.feature_encoding != entry["feature_encoding"]
                    or buffer.feature_count != entry["feature_count"]):
                raise ValueError("Replay encoding/capacity differs")
            for field, destination in (("info_states", buffer._info_states), ("iterations", buffer._iterations),
                                       ("targets", buffer._targets)):
                array = np.load(files[field], mmap_mode="r", allow_pickle=False)
                if array.shape != (entry["size"], *destination.shape[1:]) or array.dtype != destination.dtype:
                    raise ValueError("Replay array shape/precision differs")
                for start in range(0, len(array), 65536):
                    end = min(start + 65536, len(array))
                    np.copyto(destination[start:end], array[start:end], casting="no")
            buffer._size, buffer._add_calls = entry["size"], entry["add_calls"]
        for key in EXTRA_ATTRIBUTES:
            setattr(solver, key, payload["extra_attributes"][key])
        for net, mode in zip(solver._advantage_networks, payload["advantage_training_modes"]):
            net.train(mode)
        solver._policy_network.train(payload["policy_training_mode"])
        # Loading weights must preserve the live scripted and transport views.
        for eager, scripted in zip(solver._eager_advantages, solver._scripted_advantages):
            if any(eager.state_dict()[k].data_ptr() != scripted.state_dict()[k].data_ptr()
                   for k in eager.state_dict()):
                raise RuntimeError("Restoration broke shared inference weights")
        # Do this LAST: model construction and verification must not advance
        # the resumed learner/replay random streams.
        import random
        rng = payload["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        return solver, context
    except BaseException:
        solver.close()
        raise
