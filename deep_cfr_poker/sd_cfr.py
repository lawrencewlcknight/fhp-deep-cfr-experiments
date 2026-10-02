"""Single Deep CFR archives and trajectory-sampled play for FHP.

SD-CFR removes Deep CFR's separately fitted average-policy network.  Instead,
it retains each player's advantage network after every CFR update.  A playable
agent samples one historical network per player at the start of a trajectory,
with probability proportional to the CFR averaging weight, and uses that
network for the complete trajectory.

For small perfect-recall games, this module can also reconstruct
the equivalent behavioural average exactly.  At information set ``I`` the
iteration strategy is weighted by both its CFR iteration weight and its own
reach probability.  This distinction is essential: a pointwise mean of action
probabilities is not, in general, the CFR average strategy.
Full-tree reconstruction is explicitly blocked for universal poker/FHP.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from open_spiel.python import policy as osp_policy

from .networks import build_network, build_shared_trunk_player_heads


SD_CFR_ARCHIVE_TYPE = "single_deep_cfr_advantage_archive"
SD_CFR_ARCHIVE_VERSION = 1
InfoKey = Tuple[int, str]
OwnAction = Tuple[InfoKey, int]


@dataclass(frozen=True)
class AdvantageSnapshot:
    """One player's fitted advantage network after a CFR update."""

    player: int
    iteration: int
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class InformationSetRecord:
    """Static information required for exact reach-weighted averaging."""

    player: int
    key: InfoKey
    info_state: np.ndarray
    legal_actions: Tuple[int, ...]
    own_action_sequence: Tuple[OwnAction, ...]


def _clone_state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


class SDCFRArchive:
    """Player-indexed archive of historical advantage networks.

    The archive is deliberately independent from :class:`DeepCFRSolver` so it
    can be loaded for play or analysis without restoring replay buffers and
    optimiser state.
    """

    def __init__(
        self,
        *,
        game_name: str,
        num_players: int,
        num_actions: int,
        embedding_size: int,
        advantage_network_type: str,
        advantage_network_layers: Sequence[int],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.game_name = str(game_name)
        self.num_players = int(num_players)
        self.num_actions = int(num_actions)
        self.embedding_size = int(embedding_size)
        self.advantage_network_type = str(advantage_network_type)
        self.advantage_network_layers = tuple(
            int(width) for width in advantage_network_layers
        )
        self.metadata = dict(metadata or {})
        from .fhp_features import encoder_from_metadata
        self.feature_encoder = encoder_from_metadata(self.metadata.get("feature_encoder"))
        self._entries: Dict[int, List[AdvantageSnapshot]] = {
            player: [] for player in range(self.num_players)
        }

    @classmethod
    def from_solver(
        cls,
        solver,
        *,
        game_name: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "SDCFRArchive":
        metadata = dict(metadata or {})
        metadata["game_string"] = str(solver._game)
        if game_name is None:
            game_name = solver._game.get_type().short_name
            if game_name == "universal_poker":
                game_name = "FHP"
        return cls(
            game_name=game_name,
            num_players=solver._num_players,
            num_actions=solver._num_actions,
            embedding_size=solver._embedding_size,
            advantage_network_type=solver._advantage_network_type,
            advantage_network_layers=solver._advantage_network_layers,
            metadata=metadata,
        )

    @property
    def entries_by_player(self) -> Mapping[int, Tuple[AdvantageSnapshot, ...]]:
        return {
            player: tuple(entries)
            for player, entries in self._entries.items()
        }

    def capture_from_solver(self, solver, player: int, iteration: int) -> None:
        """Copies one newly fitted player network into the archive."""
        player = int(player)
        iteration = int(iteration)
        if player not in self._entries:
            raise ValueError(f"Invalid player {player}; expected 0..{self.num_players - 1}")
        entries = self._entries[player]
        if entries and iteration <= entries[-1].iteration:
            raise ValueError(
                "SD-CFR snapshots must be captured in strictly increasing "
                f"iteration order for player {player}: got {iteration} after "
                f"{entries[-1].iteration}"
            )
        entries.append(
            AdvantageSnapshot(
                player=player,
                iteration=iteration,
                state_dict=_clone_state_dict_to_cpu(
                    solver._advantage_networks[player].state_dict()
                ),
            )
        )

    def validate(self, *, require_aligned_players: bool = True) -> None:
        for player in range(self.num_players):
            entries = self._entries[player]
            if not entries:
                raise ValueError(f"SD-CFR archive has no snapshots for player {player}")
            iterations = [entry.iteration for entry in entries]
            if any(entry.player != player or entry.iteration < 1 for entry in entries):
                raise ValueError("Invalid player or non-positive snapshot iteration")
            if any(a >= b for a, b in zip(iterations, iterations[1:])):
                raise ValueError(
                    f"Player {player} archive iterations are not strictly increasing"
                )
        if require_aligned_players:
            reference = [entry.iteration for entry in self._entries[0]]
            for player in range(1, self.num_players):
                candidate = [entry.iteration for entry in self._entries[player]]
                if candidate != reference:
                    raise ValueError(
                        "SD-CFR archive players do not contain the same iteration "
                        f"schedule: player 0={reference}, player {player}={candidate}"
                    )

    def information_state(self, state, player):
        if self.feature_encoder is not None:
            return self.feature_encoder.information_state(state, player)
        return state.information_state_tensor(player)

    def validate_game(self, game) -> None:
        """Reject archives with incompatible input, action or game contracts."""
        dimensions = (game.num_players(), game.num_distinct_actions(),
                      len(self.information_state(game.new_initial_state(), 0)))
        if dimensions != (self.num_players, self.num_actions, self.embedding_size):
            raise ValueError("SD-CFR archive dimensions do not match the game")
        expected = self.metadata.get("game_string")
        if expected is not None and expected != str(game):
            raise ValueError("SD-CFR archive game contract does not match the game")

    def to_payload(self) -> dict:
        self.validate(require_aligned_players=True)
        return {
            "version": SD_CFR_ARCHIVE_VERSION,
            "type": SD_CFR_ARCHIVE_TYPE,
            "game_name": self.game_name,
            "num_players": self.num_players,
            "num_actions": self.num_actions,
            "embedding_size": self.embedding_size,
            "advantage_network_type": self.advantage_network_type,
            "advantage_network_layers": self.advantage_network_layers,
            "metadata": copy.deepcopy(self.metadata),
            "entries_by_player": {
                int(player): [
                    {
                        "player": entry.player,
                        "iteration": entry.iteration,
                        "state_dict": _clone_state_dict_to_cpu(entry.state_dict),
                    }
                    for entry in entries
                ]
                for player, entries in self._entries.items()
            },
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)
        return path

    @classmethod
    def load(cls, path, *, map_location: str = "cpu") -> "SDCFRArchive":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("type") != SD_CFR_ARCHIVE_TYPE:
            raise ValueError(f"{path} is not an SD-CFR advantage archive")
        if int(payload.get("version", -1)) != SD_CFR_ARCHIVE_VERSION:
            raise ValueError(
                f"Unsupported SD-CFR archive version {payload.get('version')}"
            )
        archive = cls(
            game_name=payload["game_name"],
            num_players=payload["num_players"],
            num_actions=payload["num_actions"],
            embedding_size=payload["embedding_size"],
            advantage_network_type=payload["advantage_network_type"],
            advantage_network_layers=payload["advantage_network_layers"],
            metadata=payload.get("metadata", {}),
        )
        for player_raw, rows in payload["entries_by_player"].items():
            player = int(player_raw)
            for row in rows:
                archive._entries[player].append(
                    AdvantageSnapshot(
                        player=player,
                        iteration=int(row["iteration"]),
                        state_dict=_clone_state_dict_to_cpu(row["state_dict"]),
                    )
                )
        archive.validate(require_aligned_players=True)
        return archive


def _build_snapshot_network(archive: SDCFRArchive, entry: AdvantageSnapshot):
    # Network construction initialises weights before loading. Do not allow
    # inference/reloading to perturb the training or simulator RNG streams.
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            return _initialise_snapshot_network(archive, entry)
    finally:
        np.random.set_state(numpy_state)


def _initialise_snapshot_network(archive: SDCFRArchive, entry: AdvantageSnapshot):
    if archive.advantage_network_type == "shared_trunk_player_heads":
        networks = build_shared_trunk_player_heads(
            archive.embedding_size,
            archive.advantage_network_layers,
            archive.num_actions,
            archive.num_players,
        )
        network = networks[entry.player]
    else:
        network = build_network(
            archive.advantage_network_type,
            archive.embedding_size,
            archive.advantage_network_layers,
            archive.num_actions,
        )
    network.load_state_dict(entry.state_dict, strict=True)
    network.eval()
    return network


def regret_matching_probabilities(
    raw_advantages: Sequence[float],
    legal_actions: Sequence[int],
    num_actions: int,
) -> np.ndarray:
    """Applies the solver's legal-action regret-matching rule."""
    legal_actions = tuple(int(action) for action in legal_actions)
    probs = np.zeros(int(num_actions), dtype=np.float64)
    positives = np.maximum(np.asarray(raw_advantages, dtype=np.float64), 0.0)
    normalizer = float(np.sum(positives[list(legal_actions)]))
    if normalizer > 0.0:
        probs[list(legal_actions)] = positives[list(legal_actions)] / normalizer
    else:
        probs[list(legal_actions)] = 1.0 / len(legal_actions)
    return probs


def build_information_set_catalog(game) -> Dict[InfoKey, InformationSetRecord]:
    """Enumerates information sets and each player's preceding action sequence.

    Perfect recall implies that every history in one information set has the
    same sequence of that player's prior information sets and actions.  The
    function verifies this invariant while traversing the game tree.
    """
    if game.get_type().short_name == "universal_poker":
        raise ValueError(
            "Full-tree SD-CFR reconstruction is disabled for FHP/universal poker. "
            "Use SampledSDCFRPolicy and resample_episode() once per hand."
        )
    num_players = int(game.num_players())
    empty_history = tuple(() for _ in range(num_players))
    stack = [(game.new_initial_state(), empty_history)]
    catalog: Dict[InfoKey, InformationSetRecord] = {}

    while stack:
        state, own_histories = stack.pop()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for action, _probability in state.chance_outcomes():
                stack.append((state.child(action), own_histories))
            continue

        player = int(state.current_player())
        info_string = str(state.information_state_string(player))
        key: InfoKey = (player, info_string)
        info_state = np.asarray(
            state.information_state_tensor(player), dtype=np.float32
        )
        legal_actions = tuple(int(action) for action in state.legal_actions(player))
        sequence = tuple(own_histories[player])
        previous = catalog.get(key)
        if previous is None:
            catalog[key] = InformationSetRecord(
                player=player,
                key=key,
                info_state=info_state,
                legal_actions=legal_actions,
                own_action_sequence=sequence,
            )
        else:
            if previous.legal_actions != legal_actions:
                raise ValueError(f"Legal actions differ within information set {key}")
            if previous.own_action_sequence != sequence:
                raise ValueError(
                    "Game does not satisfy the perfect-recall action-sequence "
                    f"invariant at information set {key}"
                )
            if not np.array_equal(previous.info_state, info_state):
                raise ValueError(f"Information-state tensor differs within {key}")

        for action in legal_actions:
            next_histories = list(own_histories)
            next_histories[player] = sequence + ((key, int(action)),)
            stack.append((state.child(action), tuple(next_histories)))

    return catalog


def _snapshot_strategy_table(
    archive: SDCFRArchive,
    entry: AdvantageSnapshot,
    player_records: Sequence[InformationSetRecord],
) -> Dict[InfoKey, np.ndarray]:
    network = _build_snapshot_network(archive, entry)
    info_states = np.asarray(
        [record.info_state for record in player_records], dtype=np.float32
    )
    with torch.no_grad():
        raw_batch = network(torch.as_tensor(info_states, dtype=torch.float32))
        raw_batch = raw_batch.detach().cpu().numpy()
    return {
        record.key: regret_matching_probabilities(
            raw,
            record.legal_actions,
            archive.num_actions,
        )
        for record, raw in zip(player_records, raw_batch)
    }


def _own_reach(
    record: InformationSetRecord,
    strategy_table: Mapping[InfoKey, np.ndarray],
) -> float:
    reach = 1.0
    for predecessor_key, action in record.own_action_sequence:
        reach *= float(strategy_table[predecessor_key][action])
    return reach


class TabularSDCFRPolicy(osp_policy.Policy):
    """OpenSpiel policy backed by an exact SD-CFR behavioural average."""

    def __init__(self, game, probabilities: Mapping[InfoKey, np.ndarray]) -> None:
        super().__init__(game, list(range(game.num_players())))
        self._game = game
        self._probabilities = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in probabilities.items()
        }

    def action_probabilities(self, state, player_id=None):
        player = int(state.current_player() if player_id is None else player_id)
        legal_actions = tuple(int(action) for action in state.legal_actions(player))
        if not legal_actions:
            return {}
        key = (player, str(state.information_state_string(player)))
        probs = self._probabilities.get(key)
        if probs is None:
            return {action: 1.0 / len(legal_actions) for action in legal_actions}
        legal_mass = float(np.sum(probs[list(legal_actions)]))
        if legal_mass <= 0.0:
            return {action: 1.0 / len(legal_actions) for action in legal_actions}
        return {
            action: float(probs[action] / legal_mass)
            for action in legal_actions
        }


def _iteration_weight(iteration: int, weighting: str) -> float:
    weighting = str(weighting).lower()
    if weighting == "linear":
        return float(iteration)
    if weighting == "uniform":
        return 1.0
    raise ValueError("weighting must be 'linear' or 'uniform'")


def exact_average_policies_at_checkpoints(
    game,
    archive: SDCFRArchive,
    checkpoints: Iterable[int],
    *,
    weighting: str = "linear",
) -> Dict[int, TabularSDCFRPolicy]:
    """Builds exact reach-weighted SD-CFR policies for archive prefixes.

    Every historical network is evaluated once.  Numerators and denominators
    are accumulated incrementally, so requesting many temporal checkpoints is
    substantially cheaper than reconstructing each prefix independently.
    """
    archive.validate(require_aligned_players=True)
    archive.validate_game(game)
    checkpoints = sorted({int(value) for value in checkpoints})
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("checkpoints must contain positive iteration numbers")

    catalog = build_information_set_catalog(game)
    records_by_player = {
        player: sorted(
            (record for record in catalog.values() if record.player == player),
            key=lambda record: record.key[1],
        )
        for player in range(archive.num_players)
    }
    numerators = {
        key: np.zeros(archive.num_actions, dtype=np.float64) for key in catalog
    }
    denominators = {key: 0.0 for key in catalog}
    entry_positions = {player: 0 for player in range(archive.num_players)}
    outputs: Dict[int, TabularSDCFRPolicy] = {}

    for checkpoint in checkpoints:
        for player in range(archive.num_players):
            entries = archive._entries[player]
            position = entry_positions[player]
            while position < len(entries) and entries[position].iteration <= checkpoint:
                entry = entries[position]
                strategies = _snapshot_strategy_table(
                    archive, entry, records_by_player[player]
                )
                weight = _iteration_weight(entry.iteration, weighting)
                for record in records_by_player[player]:
                    reach_weight = weight * _own_reach(record, strategies)
                    numerators[record.key] += reach_weight * strategies[record.key]
                    denominators[record.key] += reach_weight
                position += 1
            entry_positions[player] = position

        if any(entry_positions[player] == 0 for player in range(archive.num_players)):
            raise ValueError(
                f"Archive contains no complete player strategy by checkpoint {checkpoint}"
            )

        probabilities: Dict[InfoKey, np.ndarray] = {}
        for key, record in catalog.items():
            denominator = denominators[key]
            if denominator > 0.0:
                probs = numerators[key] / denominator
            else:
                probs = np.zeros(archive.num_actions, dtype=np.float64)
                probs[list(record.legal_actions)] = 1.0 / len(record.legal_actions)
            probabilities[key] = probs.copy()
        outputs[checkpoint] = TabularSDCFRPolicy(game, probabilities)

    return outputs


def exact_average_policy(
    game,
    archive: SDCFRArchive,
    *,
    checkpoint: Optional[int] = None,
    weighting: str = "linear",
) -> TabularSDCFRPolicy:
    """Returns one exact SD-CFR behavioural average policy."""
    archive.validate(require_aligned_players=True)
    if checkpoint is None:
        checkpoint = min(
            entries[-1].iteration for entries in archive._entries.values()
        )
    return exact_average_policies_at_checkpoints(
        game, archive, [int(checkpoint)], weighting=weighting
    )[int(checkpoint)]


class SampledSDCFRPolicy(osp_policy.Policy):
    """Playable SD-CFR trajectory policy with one sampled model per player.

    Create a fresh instance, or call :meth:`resample_episode`, at the start of
    every game.  The selected model is then held fixed for the entire game,
    which preserves the historical strategy's own-reach correlation.
    """

    def __init__(
        self,
        game,
        archive: SDCFRArchive,
        *,
        seed: Optional[int] = None,
        weighting: str = "linear",
        checkpoint: Optional[int] = None,
    ) -> None:
        super().__init__(game, list(range(game.num_players())))
        archive.validate(require_aligned_players=True)
        archive.validate_game(game)
        self._game = game
        self._archive = archive
        self._rng = np.random.default_rng(seed)
        self._weighting = str(weighting)
        self._checkpoint = checkpoint
        self._networks: Dict[int, torch.nn.Module] = {}
        self.selected_iterations: Dict[int, int] = {}
        self.resample_episode()

    def resample_episode(self) -> None:
        self._networks = {}
        self.selected_iterations = {}
        for player in range(self._archive.num_players):
            entries = [
                entry
                for entry in self._archive._entries[player]
                if self._checkpoint is None or entry.iteration <= self._checkpoint
            ]
            if not entries:
                raise ValueError(
                    f"No player {player} snapshots at checkpoint {self._checkpoint}"
                )
            weights = np.asarray(
                [_iteration_weight(entry.iteration, self._weighting) for entry in entries],
                dtype=np.float64,
            )
            weights /= weights.sum()
            selected = entries[int(self._rng.choice(len(entries), p=weights))]
            self._networks[player] = _build_snapshot_network(self._archive, selected)
            self.selected_iterations[player] = int(selected.iteration)

    def action_probabilities(self, state, player_id=None):
        player = int(state.current_player() if player_id is None else player_id)
        legal_actions = tuple(int(action) for action in state.legal_actions(player))
        if not legal_actions:
            return {}
        info_state = np.asarray(
            self._archive.information_state(state, player), dtype=np.float32
        )
        with torch.no_grad():
            raw = self._networks[player](
                torch.as_tensor(info_state[None, :], dtype=torch.float32)
            )[0].detach().cpu().numpy()
        probs = regret_matching_probabilities(
            raw, legal_actions, self._archive.num_actions
        )
        return {action: float(probs[action]) for action in legal_actions}


class HistoricalSDCFRPolicy(osp_policy.Policy):
    """Policy using explicitly selected historical networks.

    This is useful for exact tests and head-to-head analysis of particular
    iteration strategies. ``iterations_by_player`` may contain only the
    players for which this policy instance will be queried.
    """

    def __init__(
        self,
        game,
        archive: SDCFRArchive,
        iterations_by_player: Mapping[int, int],
    ) -> None:
        super().__init__(game, list(range(game.num_players())))
        archive.validate(require_aligned_players=True)
        archive.validate_game(game)
        self._game = game
        self._archive = archive
        self.selected_iterations = {
            int(player): int(iteration)
            for player, iteration in iterations_by_player.items()
        }
        self._networks: Dict[int, torch.nn.Module] = {}
        for player, iteration in self.selected_iterations.items():
            matches = [
                entry
                for entry in archive._entries[player]
                if entry.iteration == iteration
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one player {player} snapshot at iteration "
                    f"{iteration}, found {len(matches)}"
                )
            self._networks[player] = _build_snapshot_network(archive, matches[0])

    def action_probabilities(self, state, player_id=None):
        player = int(state.current_player() if player_id is None else player_id)
        if player not in self._networks:
            raise ValueError(
                f"HistoricalSDCFRPolicy has no selected network for player {player}"
            )
        legal_actions = tuple(int(action) for action in state.legal_actions(player))
        if not legal_actions:
            return {}
        info_state = np.asarray(
            self._archive.information_state(state, player), dtype=np.float32
        )
        with torch.no_grad():
            raw = self._networks[player](
                torch.as_tensor(info_state[None, :], dtype=torch.float32)
            )[0].detach().cpu().numpy()
        probs = regret_matching_probabilities(
            raw, legal_actions, self._archive.num_actions
        )
        return {action: float(probs[action]) for action in legal_actions}


__all__ = [
    "AdvantageSnapshot",
    "InformationSetRecord",
    "HistoricalSDCFRPolicy",
    "SDCFRArchive",
    "SampledSDCFRPolicy",
    "TabularSDCFRPolicy",
    "build_information_set_catalog",
    "exact_average_policies_at_checkpoints",
    "exact_average_policy",
    "regret_matching_probabilities",
]
