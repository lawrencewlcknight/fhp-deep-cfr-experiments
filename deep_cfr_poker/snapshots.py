"""Lightweight policy snapshots for trained Deep CFR policies.

A *policy snapshot* contains only what is needed to evaluate a learned average
policy: the policy network state dict, the network architecture, and a small
amount of metadata. It does **not** contain replay buffers, advantage networks,
optimiser state or RNG state.

Snapshots are deliberately decoupled from the solver class so that downstream
analysis code never has to instantiate a full :class:`DeepCFRSolver`. They are
also cheap to load (a single small ``torch.load``), which matters when an
analysis loops over many checkpoints and many seeds.

Snapshot dict schema (``version: 2``)::

    {
      "version": 2,
      "type": "deep_cfr_policy_snapshot",
      "experiment_name": str,
      "seed": int,
      "checkpoint_iteration": int,            # the schedule milestone
      "internal_iteration": int,              # solver._iteration at save time
      "stage_label": str,
      "game": str,                            # OpenSpiel game name
      "policy_state_dict": dict[str, Tensor],
      "policy_network_layers": tuple[int, ...],
      "num_actions": int,
      "embedding_size": int,
      "solver_config": dict,                  # config kwargs used to train
    }

For backwards compatibility, :class:`LoadedPolicy` also accepts:

* The thesis notebook's ``version: 1`` snapshots (same fields except
  ``solver_config`` may be missing).
* Legacy *full* checkpoints produced by either the notebook or the previous
  package (anything with a ``policy_state_dict`` and from which we can infer
  the MLP architecture).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from open_spiel.python import policy as osp_policy

from .networks import MLP


_LOGGER = logging.getLogger(__name__)

POLICY_SNAPSHOT_TYPE = "deep_cfr_policy_snapshot"
POLICY_SNAPSHOT_VERSION = 2


@dataclass
class SnapshotMetadata:
    """Tabular description of a snapshot file on disk."""

    path: Path
    seed: str
    iteration: int
    kind: str  # "policy_snapshot" | "full_checkpoint" | "legacy_full_checkpoint"


# Filename patterns. ``seed`` is optional in the legacy form because the very
# old notebook produced single-seed files without a seed in the path.
_SNAPSHOT_FILENAME_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "policy_snapshot",
        re.compile(
            r"(?:.*?)seed_(?P<seed>\d+)_policy_snapshot_(?P<iteration>\d+)_iters\.pt$"
        ),
    ),
    (
        "policy_snapshot",
        re.compile(r"^seed_(?P<seed>\d+)_iter_(?P<iteration>\d+)_snapshot\.pt$"),
    ),
    (
        "full_checkpoint",
        re.compile(r"(?:.*?)seed_(?P<seed>\d+)_ckpt_(?P<iteration>\d+)_iters\.pt$"),
    ),
    (
        "full_checkpoint",
        re.compile(r"^seed_(?P<seed>\d+)_iter_(?P<iteration>\d+)_full\.pt$"),
    ),
)


def parse_snapshot_filename(path: Path) -> Optional[SnapshotMetadata]:
    """Returns :class:`SnapshotMetadata` if ``path`` looks like a snapshot."""
    name = Path(path).name
    for kind, pattern in _SNAPSHOT_FILENAME_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        seed = match.groupdict().get("seed", "legacy")
        return SnapshotMetadata(
            path=Path(path).resolve(),
            seed=str(seed),
            iteration=int(match.group("iteration")),
            kind=kind,
        )
    return None


def discover_snapshots(directory) -> list[SnapshotMetadata]:
    """Walks ``directory`` and returns metadata for every snapshot found.

    Both flat layouts (everything in one directory) and the
    ``snapshots/seed_<seed>_iter_<iter>_snapshot.pt`` layout used by the
    package experiment runners are recognised.
    """
    directory = Path(directory)
    out: list[SnapshotMetadata] = []
    if not directory.exists():
        return out
    for path in sorted(directory.rglob("*.pt")):
        parsed = parse_snapshot_filename(path)
        if parsed is not None:
            out.append(parsed)
    return out


def package_snapshot_filename(seed: int, iteration: int) -> str:
    """Returns the package-canonical filename for a policy snapshot."""
    return f"seed_{int(seed)}_iter_{int(iteration)}_snapshot.pt"


def package_full_checkpoint_filename(seed: int, iteration: int) -> str:
    """Returns the package-canonical filename for a full checkpoint."""
    return f"seed_{int(seed)}_iter_{int(iteration)}_full.pt"


def save_policy_snapshot(
    solver,
    path,
    *,
    seed: int,
    target_iteration: int,
    stage_label: str = "",
    experiment_name: str = "",
    game_name: str = "FHP",
    solver_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes a :data:`POLICY_SNAPSHOT_TYPE` snapshot to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": POLICY_SNAPSHOT_VERSION,
        "type": POLICY_SNAPSHOT_TYPE,
        "experiment_name": str(experiment_name),
        "seed": int(seed),
        "checkpoint_iteration": int(target_iteration),
        "internal_iteration": int(getattr(solver, "_iteration", 0)),
        "stage_label": str(stage_label),
        "game": str(game_name),
        "policy_state_dict": solver._policy_network.state_dict(),
        "policy_network_layers": tuple(_infer_solver_policy_layers(solver)),
        "num_actions": int(solver._num_actions),
        "embedding_size": int(solver._embedding_size),
        "solver_config": dict(solver_config or {}),
    }
    torch.save(payload, path)
    _LOGGER.info("Wrote policy snapshot: %s", path)
    return path


def _infer_solver_policy_layers(solver) -> Tuple[int, ...]:
    """Best-effort introspection of the policy MLP's hidden sizes."""
    layers = []
    try:
        for layer in solver._policy_network.model:
            layers.append(int(layer._weight.shape[0]))
    except Exception:  # pragma: no cover - defensive
        return ()
    # Drop the output layer; what remains are the hidden sizes.
    return tuple(layers[:-1]) if len(layers) > 1 else tuple(layers)


def _infer_mlp_architecture_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, list, int]:
    """Infers (input_size, hidden_sizes, output_size) from MLP weight shapes."""
    pattern = re.compile(r"model\.(\d+)\._weight$")
    weights = []
    for key, value in state_dict.items():
        match = pattern.search(key)
        if match:
            weights.append((int(match.group(1)), tuple(value.shape)))
    if not weights:
        raise ValueError(
            "Could not infer MLP architecture from policy_state_dict: "
            "no entries matching 'model.{i}._weight' were found."
        )
    weights.sort(key=lambda item: item[0])
    input_size = weights[0][1][1]
    out_sizes = [shape[0] for _, shape in weights]
    output_size = out_sizes[-1]
    hidden_sizes = out_sizes[:-1]
    return int(input_size), list(hidden_sizes), int(output_size)


class LoadedPolicy(osp_policy.Policy):
    """An OpenSpiel :class:`policy.Policy` backed by a saved policy snapshot.

    Loads from any of the supported snapshot formats:

    * the package format produced by :func:`save_policy_snapshot`;
    * the thesis notebook ``version: 1`` snapshot;
    * a legacy full checkpoint that contains a ``policy_state_dict`` (the rest
      of the checkpoint is ignored).

    The policy is evaluated in inference mode on CPU. Action probabilities are
    obtained by softmaxing over **legal** actions only, matching the inference
    path used inside :class:`DeepCFRSolver`.
    """

    def __init__(self, game, snapshot_path, map_location: str = "cpu") -> None:
        super().__init__(game, list(range(game.num_players())))
        self._game = game
        self._path = Path(snapshot_path)
        ckpt = torch.load(self._path, map_location=map_location, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Snapshot file {self._path} is not a torch.save'd dict."
            )

        if "policy_state_dict" not in ckpt:
            raise ValueError(
                f"Snapshot file {self._path} does not contain 'policy_state_dict'."
            )

        state_dict = ckpt["policy_state_dict"]
        try:
            input_size, hidden_sizes, output_size = (
                _infer_mlp_architecture_from_state_dict(state_dict)
            )
        except ValueError:
            # Fall back to declared architecture if inference fails.
            hidden_sizes = list(ckpt.get("policy_network_layers", []))
            input_size = int(
                ckpt.get("embedding_size") or game.new_initial_state().information_state_tensor(0).__len__()
            )
            output_size = int(ckpt.get("num_actions") or game.num_distinct_actions())

        # Best-effort cross-check against any declared metadata.
        declared = ckpt.get("policy_network_layers")
        if declared and tuple(declared) != tuple(hidden_sizes):
            _LOGGER.warning(
                "Snapshot %s declares policy_network_layers=%s but the saved "
                "weights imply %s. Using inferred sizes.",
                self._path,
                tuple(declared),
                tuple(hidden_sizes),
            )

        self._policy_network = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )
        self._policy_network.load_state_dict(state_dict, strict=True)
        self._policy_network.eval()

        self._meta: Dict[str, Any] = {
            "version": int(ckpt.get("version", 0)),
            "type": str(ckpt.get("type", "unknown")),
            "experiment_name": str(ckpt.get("experiment_name", "")),
            "seed": ckpt.get("seed"),
            "checkpoint_iteration": ckpt.get("checkpoint_iteration"),
            "internal_iteration": ckpt.get("internal_iteration"),
            "stage_label": ckpt.get("stage_label", ""),
            "game": ckpt.get("game", ""),
            "policy_network_layers": tuple(hidden_sizes),
            "num_actions": int(output_size),
            "embedding_size": int(input_size),
            "solver_config": dict(ckpt.get("solver_config", {})),
            "path": str(self._path),
        }
        self._num_actions = int(output_size)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._meta)

    @property
    def checkpoint_iteration(self) -> Optional[int]:
        value = self._meta.get("checkpoint_iteration")
        return None if value is None else int(value)

    @property
    def seed(self) -> Optional[str]:
        value = self._meta.get("seed")
        return None if value is None else str(value)

    def action_probabilities(self, state, player_id=None) -> Dict[int, float]:
        cur_player = state.current_player() if player_id is None else player_id
        legal_actions = state.legal_actions(cur_player)
        if not legal_actions:
            return {}
        info_state_vector = np.asarray(
            state.information_state_tensor(cur_player), dtype=np.float32
        )
        if info_state_vector.ndim == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        with torch.no_grad():
            logits = self._policy_network(
                torch.as_tensor(info_state_vector, dtype=torch.float32)
            )[0]
            legal_logits = logits[legal_actions]
            legal_probs = torch.softmax(legal_logits, dim=-1).cpu().numpy()
        return {
            int(action): float(prob)
            for action, prob in zip(legal_actions, legal_probs)
        }
