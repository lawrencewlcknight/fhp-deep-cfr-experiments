"""Bounded-memory, lossless, append-only SD-CFR strategy storage.

Every player's post-update weights are captured, without changing RNG state.
An immutable checkpoint manifest references a prefix of shared NumPy chunks;
neither replay nor optimiser states are retained. All arrays remain float32.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .networks import build_network
from .sd_cfr import regret_matching_probabilities


FORMAT = "sd_cfr_chunked_uniform_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class DiskSDCFRArchive:
    def __init__(self, solver, root, *, chunk_iterations=128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if list(self.root.glob("chunk_*.npy")):
            raise ValueError("Refusing to overwrite an existing archive")
        if chunk_iterations < 1 or solver._num_players != 2:
            raise ValueError("Expected positive chunk size and two players")
        self.chunk_iterations = int(chunk_iterations)
        self.metadata = dict(solver.archive.metadata)
        state = solver._advantage_networks[0].state_dict()
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("Archive requires float32 weights (no implicit conversion)")
        self.layout = [dict(key=key, shape=list(value.shape), size=value.numel())
                       for key, value in state.items()]
        self.contract = dict(format=FORMAT, game_string=str(solver._game),
                             num_players=solver._num_players, num_actions=solver._num_actions,
                             embedding_size=solver._embedding_size,
                             network_type=solver._advantage_network_type,
                             network_layers=list(solver._advantage_network_layers),
                             layout=self.layout, weighting="uniform", metadata=self.metadata)
        self.chunks = []
        self.pending = []
        self.partial = []
        self.count = 0

    def capture_from_solver(self, solver, player, iteration):
        if int(player) != len(self.partial) or int(iteration) != self.count + 1:
            raise ValueError("Expected every player update in contiguous iteration order")
        state = solver._advantage_networks[player].state_dict()
        if list(state) != [row["key"] for row in self.layout]:
            raise ValueError("Network layout changed during training")
        flat = np.concatenate([
            state[row["key"]].detach().cpu().numpy().reshape(-1) for row in self.layout
        ])
        if not np.isfinite(flat).all():
            raise ValueError("Non-finite advantage network; refusing to archive an invalid policy")
        self.partial.append(flat)
        if len(self.partial) == 2:
            self.pending.append(np.stack(self.partial))
            self.partial = []
            self.count += 1
            if len(self.pending) >= self.chunk_iterations:
                self.flush()

    def flush(self):
        if self.partial:
            raise ValueError("Cannot checkpoint halfway through a player pair")
        if not self.pending:
            return
        start = self.count - len(self.pending) + 1
        path = self.root / f"chunk_{start:08d}_{self.count:08d}.npy"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.stack(self.pending), allow_pickle=False)
        temporary.replace(path)
        self.chunks.append(dict(path=path.name, first_iteration=start,
                                last_iteration=self.count, sha256=sha256(path),
                                size_bytes=path.stat().st_size))
        self.pending.clear()

    def validate(self, **_kwargs):
        if self.count < 1 or self.partial:
            raise ValueError("Incomplete SD-CFR archive")

    def checkpoint(self, path):
        self.validate()
        self.flush()
        # Manifests live next to chunks, making the complete archive relocatable.
        path = Path(path)
        if path.parent.resolve() != self.root.resolve():
            raise ValueError("Checkpoint manifests must share the archive directory")
        write_json(path, {**self.contract, "completed_iterations": self.count,
                          "chunks": list(self.chunks)})
        return path


class DiskArchiveReader:
    def __init__(self, path, game, *, verify=True):
        self.path = Path(path)
        self.contract = json.loads(self.path.read_text())
        c = self.contract
        if c.get("format") != FORMAT or c.get("weighting") != "uniform":
            raise ValueError("Unsupported SD-CFR checkpoint")
        if c["game_string"] != str(game):
            raise ValueError("SD-CFR checkpoint game mismatch")
        from .fhp_features import encoder_from_metadata
        self.feature_encoder = encoder_from_metadata(c.get("metadata", {}).get("feature_encoder"))
        if c["embedding_size"] != len(self.information_state(game.new_initial_state(), 0)):
            raise ValueError("SD-CFR archive input representation/dimensions mismatch")
        self.count = int(c["completed_iterations"])
        self.width = sum(row["size"] for row in c["layout"])
        self.chunks = c["chunks"]
        self._maps = OrderedDict()
        expected = 1
        for chunk in self.chunks:
            file = self.path.parent / chunk["path"]
            if file.parent.resolve() != self.path.parent.resolve():
                raise ValueError("Archive chunk must be adjacent to manifest")
            if chunk["first_iteration"] != expected:
                raise ValueError("Missing or overlapping historical strategies")
            expected = chunk["last_iteration"] + 1
            if file.stat().st_size != chunk["size_bytes"] or (verify and sha256(file) != chunk["sha256"]):
                raise ValueError(f"Archive integrity mismatch: {file}")
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (
                expected - chunk["first_iteration"], 2, self.width
            ):
                raise ValueError("Archive array shape/precision mismatch")
        if expected != self.count + 1 or self.count < 1:
            raise ValueError("Incomplete historical strategy prefix")

    def information_state(self, state, player):
        if self.feature_encoder is not None:
            return self.feature_encoder.information_state(state, player)
        return state.information_state_tensor(player)

    def chunk(self, index):
        if index not in self._maps:
            self._maps[index] = np.load(self.path.parent / self.chunks[index]["path"],
                                        mmap_mode="r", allow_pickle=False)
        self._maps.move_to_end(index)
        while len(self._maps) > 4:
            self._maps.popitem(last=False)
        return self._maps[index]

    def weights(self, player, iteration):
        if player not in (0, 1) or not 1 <= iteration <= self.count:
            raise ValueError("Player or iteration outside checkpoint")
        for index, chunk in enumerate(self.chunks):
            if chunk["first_iteration"] <= iteration <= chunk["last_iteration"]:
                flat = self.chunk(index)[iteration - chunk["first_iteration"], player]
                return self.state_dict(torch.from_numpy(flat.copy()))
        raise ValueError("Missing strategy")

    def state_dict(self, flat):
        result, offset = {}, 0
        for row in self.contract["layout"]:
            result[row["key"]] = flat[..., offset:offset + row["size"]].reshape(
                *flat.shape[:-1], *row["shape"])
            offset += row["size"]
        return result

    def network(self):
        # Construction must not perturb training/simulator RNGs.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                c = self.contract
                return build_network(c["network_type"], c["embedding_size"],
                                     c["network_layers"], c["num_actions"]).eval()
        finally:
            np.random.set_state(numpy_state)


class DiskSampledPolicy:
    """Uniform historical mixture: begin_episode is REQUIRED per hand."""
    def __init__(self, reader):
        self.reader = reader
        self.networks = [reader.network(), reader.network()]
        self.selected_iterations = {}

    def begin_episode(self, *, seed):
        rng = np.random.default_rng(seed)
        self.selected_iterations = {}
        for player, network in enumerate(self.networks):
            iteration = int(rng.integers(1, self.reader.count + 1))
            self.selected_iterations[player] = iteration
            network.load_state_dict(self.reader.weights(player, iteration), strict=True)

    def action_probabilities(self, state, player_id=None):
        if not self.selected_iterations:
            raise RuntimeError("Call begin_episode once before every hand")
        player = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(player)
        if not legal:
            return {}
        info = torch.tensor(self.reader.information_state(state, player), dtype=torch.float32)
        with torch.no_grad():
            raw = self.networks[player](info.unsqueeze(0))[0].numpy()
        probs = regret_matching_probabilities(raw, legal, self.reader.contract["num_actions"])
        return {action: float(probs[action]) for action in legal}


class DiskBehaviouralPolicy:
    """Exact own-reach mixture at queried histories, without full-tree expansion.

Used for LBR's hypothetical-hand queries; never exposes the sampled hidden
historical network. Vectorises over models in bounded batches. Caches only
bounded, public information-state results, not full trees or opponent cards.
"""
    def __init__(self, reader, game, *, model_batch_size=128, cache_size=20000):
        self.reader, self.game = reader, game
        self.network = reader.network()
        self.batch_size = int(model_batch_size)
        self.cache_size = int(cache_size)
        if self.batch_size < 1 or self.cache_size < 1:
            raise ValueError("Positive batching and cache bounds required")
        self.cache = OrderedDict()

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(player)
        if not legal:
            return {}
        key = (player, state.information_state_string(player))
        if key in self.cache:
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        cursor = self.game.new_initial_state()
        features, masks, own_actions = [], [], []
        for action in state.history():
            if not cursor.is_chance_node() and cursor.current_player() == player:
                features.append(self.reader.information_state(cursor, player))
                masks.append(cursor.legal_actions_mask(player))
                own_actions.append(int(action))
            cursor.apply_action(action)
        features.append(self.reader.information_state(state, player))
        masks.append(state.legal_actions_mask(player))
        inputs = torch.tensor(np.asarray(features), dtype=torch.float32)
        mask = np.asarray(masks, dtype=np.float64)
        numerator = np.zeros(self.reader.contract["num_actions"], dtype=np.float64)
        denominator = 0.0
        with torch.no_grad():
            for chunk_index in range(len(self.reader.chunks)):
                array = self.reader.chunk(chunk_index)
                for start in range(0, len(array), self.batch_size):
                    flat = torch.from_numpy(array[start:start + self.batch_size, player].copy())
                    params = self.reader.state_dict(flat)
                    raw = torch.vmap(lambda weights: torch.func.functional_call(
                        self.network, weights, (inputs,)))(params).numpy().astype(np.float64)
                    positive = np.maximum(raw, 0.0) * mask[None, :, :]
                    normalizer = positive.sum(axis=-1, keepdims=True)
                    probabilities = np.divide(positive, normalizer,
                                              out=np.zeros_like(positive), where=normalizer > 0)
                    probabilities = np.where(normalizer > 0, probabilities,
                                             mask[None, :, :] / mask.sum(axis=-1)[None, :, None])
                    reach = np.ones(len(flat), dtype=np.float64)
                    for step, action in enumerate(own_actions):
                        reach *= probabilities[:, step, action]
                    numerator += (reach[:, None] * probabilities[:, -1]).sum(axis=0)
                    denominator += reach.sum()
        result = ({action: float(numerator[action] / denominator) for action in legal}
                  if denominator > 0 else {action: 1.0 / len(legal) for action in legal})
        self.cache[key] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return dict(result)
