"""Collision-free loaders for policy artefacts emitted by all three trainers."""

from __future__ import annotations

import pickle
from pathlib import Path
import re
import hashlib

import numpy as np
from open_spiel.python import policy
import torch
import torch.nn.functional as torch_functional

from .game import FHP_GAME_PARAMETERS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise ValueError(f"Not a supported average-policy artefact: {path}")
    game = payload.get("game")
    if isinstance(game, dict) and game.get("parameters") != dict(FHP_GAME_PARAMETERS):
        raise ValueError("Policy artefact does not use the canonical FHP contract")
    if isinstance(game, str) and game.lower() not in {"fhp", "universal_poker"}:
        raise ValueError(f"Unexpected game label: {game!r}")
    return payload


def _ordered_layers(state_dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    patterns = (
        re.compile(r"^layers\.(\d+)\.weight$"),
        re.compile(r"^model\.(\d+)\._weight$"),
    )
    for pattern in patterns:
        weights = []
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match:
                weights.append((int(match.group(1)), key, value.detach().cpu()))
        if not weights:
            continue
        layers = []
        for index, key, weight in sorted(weights):
            if ".weight" in key:
                bias_key = key.replace(".weight", ".bias")
            else:
                bias_key = key.replace("._weight", "._bias")
            if bias_key not in state_dict:
                raise ValueError(f"Missing bias for policy layer {index}")
            layers.append((weight, state_dict[bias_key].detach().cpu()))
        return layers
    raise ValueError("Unsupported policy network state-dict layout")


class LoadedCheckpointPolicy(policy.Policy):
    """Framework-light inference from UCV, VR-Deep, or Deep CFR checkpoints."""

    def __init__(self, game, path: str | Path):
        super().__init__(game, list(range(game.num_players())))
        self.path = Path(path).resolve()
        self.sha256 = sha256_file(self.path)
        self.payload = _load_payload(self.path)
        self.layers = _ordered_layers(self.payload["policy_state_dict"])
        self.metadata = {
            key: value for key, value in self.payload.items() if key != "policy_state_dict"
        }

    def _logits(self, info_state) -> torch.Tensor:
        value = torch.as_tensor(np.asarray(info_state), dtype=torch.float32)
        for index, (weight, bias) in enumerate(self.layers):
            value = torch_functional.linear(value, weight, bias)
            if index + 1 < len(self.layers):
                value = torch_functional.relu(value)
        return value

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = list(int(action) for action in state.legal_actions(player))
        if not legal:
            return {}
        with torch.no_grad():
            logits = self._logits(state.information_state_tensor(player))
            probabilities = torch.softmax(logits[legal], dim=-1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Non-finite probabilities from {self.path}")
        return {action: float(value) for action, value in zip(legal, probabilities)}


def load_checkpoint_policy(game, path: str | Path) -> LoadedCheckpointPolicy:
    return LoadedCheckpointPolicy(game, path)
