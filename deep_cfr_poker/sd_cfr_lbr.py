"""Exact, batched SD-CFR queries for the unchanged shared LBR action scorer.

No historical networks, opponent hands or rollout samples are removed. For
uniform SD-CFR, the product of behavioural probabilities along a player's own
sequence telescopes to the mean historical own reach. Evaluating that mean
directly avoids repeatedly reconstructing every intermediate distribution.
"""

from collections import OrderedDict
from itertools import combinations

import numpy as np
import torch

from fhp_evaluation.cards import cards_from_information_state
from fhp_evaluation.lbr import LocalBestResponsePolicy, _Hypothesis, FOLD, RAISE
from .sd_cfr_disk import DiskBehaviouralPolicy


class BatchedDiskBehaviouralPolicy(DiskBehaviouralPolicy):
    """All-model own-reach mixture, batched over models AND queried hands.

    Memory is bounded by model/input batching, not archive length. Float32
    neural inference and float64 regret matching/reach accumulation match the
    scalar reference. Different GEMM shapes can introduce rounding differences;
    equivalence gates must pass before a production evaluation is authorised.
    """

    def __init__(self, reader, game, *, model_batch_size=128, state_batch_size=2048,
                 cache_size=20000, device="cpu", device_cache_bytes=256 * 1024**2):
        super().__init__(reader, game, model_batch_size=model_batch_size, cache_size=cache_size)
        if device not in ("cpu", "cuda"):
            raise ValueError("Use explicit cpu or cuda; no automatic precision/device change")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA LBR requested but CUDA is unavailable; refusing CPU fallback")
        self.device = torch.device(device)
        if device == "cuda":
            # Evaluation-only process. Never enable TF32, autocast, FP16 or BF16
            # as an implicit speed/accuracy trade-off for a comparative metric.
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.network.to(self.device)
        self.device_cache_bytes = int(device_cache_bytes)
        if self.device_cache_bytes < 0:
            raise ValueError("Non-negative device cache budget required")
        self._device_weights = OrderedDict()
        self._device_weight_bytes = 0
        self.state_batch_size = int(state_batch_size)
        if self.state_batch_size < 1:
            raise ValueError("Positive state batch size required")
        self._reach_cache = OrderedDict()
        self.stats = dict(network_batches=0, input_rows=0, queries=0, cache_hits=0)

    def _weights(self, chunk_index, player, start):
        key = (chunk_index, player, start)
        if key in self._device_weights:
            self._device_weights.move_to_end(key)
            return self._device_weights[key]
        array = self.reader.chunk(chunk_index)
        value = torch.from_numpy(array[start:start + self.batch_size, player].copy()).to(self.device)
        size = value.numel() * value.element_size()
        if self.device.type == "cuda" and size <= self.device_cache_bytes:
            while self._device_weights and self._device_weight_bytes + size > self.device_cache_bytes:
                _, expired = self._device_weights.popitem(last=False)
                self._device_weight_bytes -= expired.numel() * expired.element_size()
            self._device_weights[key] = value
            self._device_weight_bytes += size
        return value

    def _remember(self, key, value):
        self._reach_cache[key] = value
        self._reach_cache.move_to_end(key)
        while len(self._reach_cache) > self.cache_size:
            self._reach_cache.popitem(last=False)

    def batch_reach_and_probabilities(self, states, player, *, include_action=True):
        """Return mean own reach and conditional action probabilities per state.

        ``include_action=False`` permits non-acting player reach queries. Own
        reach excludes chance and the other player's actions, exactly as in
        SD-CFR behavioural reconstruction and LBR's Bayesian likelihood.
        """
        player = int(player)
        if player not in (0, 1):
            raise ValueError("Expected player 0 or 1")
        states = list(states)
        n_actions = self.reader.contract["num_actions"]
        reach = np.empty(len(states), dtype=np.float64)
        probabilities = np.zeros((len(states), n_actions), dtype=np.float64)
        pending = OrderedDict()
        self.stats["queries"] += len(states)
        for index, state in enumerate(states):
            if include_action and (state.is_terminal() or state.is_chance_node()
                                   or state.current_player() != player):
                raise ValueError("Action queries require the acting player")
            key = (player, state.information_state_string(player), bool(include_action))
            cached = self._reach_cache.get(key)
            if cached is not None:
                self._reach_cache.move_to_end(key)
                reach[index], probabilities[index] = cached
                self.stats["cache_hits"] += 1
            elif key in pending:
                pending[key][1].append(index)
            else:
                pending[key] = (state, [index])
        entries = list(pending.items())
        for start in range(0, len(entries), self.state_batch_size):
            batch = entries[start:start + self.state_batch_size]
            weights, actions = self._evaluate_batch([item[1][0] for item in batch], player, include_action)
            for (key, (_, indices)), weight, distribution in zip(batch, weights, actions):
                value = (float(weight), distribution.copy())
                self._remember(key, value)
                for index in indices:
                    reach[index], probabilities[index] = value
        return reach, probabilities

    def _evaluate_batch(self, states, player, include_action):
        rows, masks, row_index, paths, ends = [], [], {}, [], []

        def add_input(state):
            features = np.asarray(self.reader.information_state(state, player), dtype=np.float32)
            mask = np.asarray(state.legal_actions_mask(player), dtype=np.float64)
            # Exact identical network inputs can share one forward pass, even
            # when different absolute suit names canonicalise to the same input.
            key = (features.tobytes(), mask.tobytes())
            if key not in row_index:
                row_index[key] = len(rows)
                rows.append(features.copy())
                masks.append(mask)
            return row_index[key]

        for state in states:
            cursor = self.game.new_initial_state()
            path = []
            for action in state.history():
                if not cursor.is_chance_node() and cursor.current_player() == player:
                    path.append((add_input(cursor), int(action)))
                cursor.apply_action(action)
            paths.append(path)
            ends.append(add_input(state) if include_action else -1)
        n_actions = self.reader.contract["num_actions"]
        denominator = np.zeros(len(states), dtype=np.float64)
        numerator = np.zeros((len(states), n_actions), dtype=np.float64)
        if not rows:
            return np.ones(len(states), dtype=np.float64), numerator
        inputs = torch.from_numpy(np.stack(rows)).to(self.device)
        mask = np.stack(masks)
        uniform = mask / mask.sum(axis=-1, keepdims=True)
        # History layout is invariant across all model chunks. Build integer
        # gathers once, not again for each of thousands of historical networks.
        steps = []
        for depth in range(max(map(len, paths), default=0)):
            active = np.asarray([i for i, path in enumerate(paths) if depth < len(path)])
            positions = np.asarray([paths[i][depth][0] for i in active])
            actions = np.asarray([paths[i][depth][1] for i in active])
            steps.append((active, positions, actions))
        self.stats["input_rows"] += len(rows)
        predict = torch.vmap(lambda weights: torch.func.functional_call(self.network, weights, (inputs,)))
        with torch.inference_mode():
            for chunk_index in range(len(self.reader.chunks)):
                array = self.reader.chunk(chunk_index)
                for start in range(0, len(array), self.batch_size):
                    flat = self._weights(chunk_index, player, start)
                    raw = predict(self.reader.state_dict(flat)).cpu().numpy().astype(np.float64)
                    if not np.isfinite(raw).all():
                        raise ValueError("Non-finite archived network prediction")
                    positive = np.maximum(raw, 0.0) * mask[None, :, :]
                    total = positive.sum(axis=-1, keepdims=True)
                    probs = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
                    probs = np.where(total > 0, probs, uniform[None, :, :])
                    own_reach = np.ones((len(flat), len(states)), dtype=np.float64)
                    for active, positions, actions in steps:
                        own_reach[:, active] *= probs[:, positions, actions]
                    denominator += own_reach.sum(axis=0)
                    if include_action:
                        numerator += (own_reach[:, :, None] * probs[:, ends, :]).sum(axis=0)
                    self.stats["network_batches"] += 1
        distributions = np.zeros_like(numerator)
        if include_action:
            np.divide(numerator, denominator[:, None], out=distributions, where=denominator[:, None] > 0)
            zero = denominator == 0
            distributions[zero] = mask[np.asarray(ends)[zero]] / mask[np.asarray(ends)[zero]].sum(axis=1, keepdims=True)
        return denominator / self.reader.count, distributions

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(player)
        if not legal:
            return {}
        _, probabilities = self.batch_reach_and_probabilities([state], player)
        return {action: float(probabilities[0, action]) for action in legal}


class ExactSDCFRLocalBestResponsePolicy(LocalBestResponsePolicy):
    """Shared LBR scorer/equity/RNG unchanged; only exact target queries batch.

    This object receives the full deterministic behavioural mixture, never the
    trajectory policy or its secretly sampled historical-network identity.
    """

    def __init__(self, game, target_policy, *, config=None):
        if not isinstance(target_policy, BatchedDiskBehaviouralPolicy):
            raise TypeError("Exact SD-CFR LBR requires the all-model batched mixture")
        super().__init__(game, target_policy, config=config)
        self._prepared_folds = {}

    def _opponent_range(self, state, responder):
        private, board = cards_from_information_state(state, responder)
        excluded = set(private + board)
        hands = tuple(combinations((c for c in range(52) if c not in excluded), 2))
        history = tuple(int(a) for a in state.history())
        opponent = 1 - responder
        slots = (2, 3) if responder == 0 else (0, 1)
        hypotheses, queries = [], []
        can_raise = RAISE in state.legal_actions(responder)
        for hand in hands:
            replacements = dict(zip(slots, hand))
            child = self.game.new_initial_state()
            chance_index = 0
            for observed in history:
                action = observed
                if child.is_chance_node():
                    action = replacements.get(chance_index, action)
                    chance_index += 1
                child.apply_action(action)
            hypotheses.append(_Hypothesis(hand, 0.0, child))
            query = child.clone()
            if can_raise:
                query.apply_action(RAISE)
                if query.is_terminal() or query.is_chance_node() or query.current_player() != opponent:
                    raise ValueError("Unexpected FHP raise transition")
            queries.append(query)
        # The responder's proposed raise cannot change the opponent's own reach.
        likelihoods, distributions = self.target_policy.batch_reach_and_probabilities(
            queries, opponent, include_action=can_raise)
        total = float(sum(likelihoods))  # Same hand order as the shared scalar LBR.
        if not np.isfinite(total) or total <= self.config.probability_tolerance:
            raise RuntimeError("Bayesian opponent range collapsed to zero mass")
        self._prepared_folds = {}
        result = []
        for hypothesis, weight, distribution in zip(hypotheses, likelihoods, distributions):
            if weight > 0.0:
                hypothesis.weight = float(weight / total)
                result.append(hypothesis)
                if can_raise:
                    # The shared _policy_probability normalises legal mass too.
                    self._prepared_folds[hypothesis.hand] = float(distribution[FOLD] / distribution.sum())
        return result

    def _raise_fold_range(self, hypotheses, responder):
        fold_probability, continuing_mass = 0.0, 0.0
        continuing = []
        for hypothesis in hypotheses:
            child = hypothesis.state.clone()
            child.apply_action(RAISE)
            probability = self._prepared_folds[hypothesis.hand]
            fold_probability += hypothesis.weight * probability
            weight = hypothesis.weight * (1.0 - probability)
            if weight > 0.0:
                continuing.append(_Hypothesis(hypothesis.hand, weight, child))
                continuing_mass += weight
        if continuing_mass > self.config.probability_tolerance:
            for hypothesis in continuing:
                hypothesis.weight /= continuing_mass
        else:
            continuing = list(hypotheses)
        return float(fold_probability), continuing
