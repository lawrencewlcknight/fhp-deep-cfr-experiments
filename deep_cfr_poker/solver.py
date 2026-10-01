"""Deep CFR solver used by the poker experiments.

This implementation follows Brown, Lerer, Gross & Sandholm (2019),
"Deep Counterfactual Regret Minimization" (https://arxiv.org/abs/1811.00164),
and is a packaged refactor of the original notebook used for the thesis
diagnostic runs.

The class extends OpenSpiel's :class:`policy.Policy` so the learned average
policy can be passed directly to OpenSpiel evaluation utilities such as
``exploitability.nash_conv``.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401 -- kept for parity with original notebook
from torch import nn

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability

from .networks import build_network, build_shared_trunk_player_heads
from .replay import (
    AdvantageMemory,
    ReplayBatch,
    ReservoirBuffer,
    StrategyMemory,
    make_advantage_buffer,
    make_strategy_buffer,
    records_to_batch,
)


_LOGGER = logging.getLogger(__name__)

# Cap how many samples ``_advantage_target_summary`` consumes per checkpoint to
# keep diagnostic cost independent of buffer capacity.
_ADVANTAGE_DIAGNOSTIC_MAX_SAMPLES = 4096


@dataclass
class SolveResult:
    """Container returned by :meth:`DeepCFRSolver.solve`.

    Using a dataclass instead of a positional tuple makes the contract explicit
    and lets the diagnostic surface grow without breaking call sites.
    """

    policy_network: Optional[nn.Module]
    advantage_losses: Dict[int, List[float]]
    policy_losses: List[Optional[float]]
    nash_conv: List[float]
    nodes_touched: List[int]
    average_policy_value: List[float]
    diagnostics: Dict[str, list] = field(default_factory=dict)


class DeepCFRSolver(policy.Policy):
    """Deep CFR solver in PyTorch.

    Args:
        game: An OpenSpiel game.
        policy_network_type: Network architecture for the average-policy
            function approximator. ``"mlp"`` preserves the baseline. Supported
            opt-in values are ``"residual_mlp"``, ``"layer_norm_mlp"``, and
            ``"residual_layer_norm_mlp"``.
        policy_network_layers: Hidden layer sizes for the average-policy MLP.
        advantage_network_type: Network architecture for each advantage
            function approximator. Uses the same supported values as
            ``policy_network_type``. The additional value
            ``"shared_trunk_player_heads"`` uses one shared advantage trunk
            with a separate action-output head for each player.
        advantage_network_layers: Hidden layer sizes for each advantage MLP.
        num_iterations: Number of Deep CFR iterations to run.
        num_traversals: Number of external-sampling traversals per iteration
            per traversing player.
        learning_rate: Initial Adam learning rate for both networks.
        learning_rate_schedule: Learning-rate schedule. Supported values are
            ``"constant"``, ``"linear_decay"``, ``"cosine_decay"``, and
            ``"step_decay"``. The default preserves the baseline behaviour.
        learning_rate_end: Terminal learning rate for decay schedules. Defaults
            to 10% of ``learning_rate`` for decay schedules.
        learning_rate_decay_rate: Multiplicative factor for ``"step_decay"``.
        learning_rate_decay_steps: CFR iterations between step decays.
        learning_rate_warmup_iterations: Optional linear warmup length.
        batch_size_advantage: If set, sample this many transitions from the
            advantage buffer per gradient step. ``None`` trains on the full
            buffer in memory.
        batch_size_strategy: As above for the average-policy buffer.
        memory_capacity: Reservoir-buffer capacity for both buffers.
        policy_network_train_steps: SGD steps per training session of the
            average-policy network.
        policy_network_train_every: Train the average-policy network every this
            many iterations in intermittent mode. Must be >= 1.
        evaluation_interval: Evaluate exploitability and diagnostics every this
            many iterations. Defaults to ``policy_network_train_every`` for
            backwards-compatible checkpoint behaviour.
        policy_training_mode: ``"intermittent"``, ``"final_only"``, or ``"disabled"``.
            Final-only mode trains the average-policy network once after CFR
            data collection finishes. Disabled mode performs no policy fitting
            and is used for standalone SD-CFR.
        collect_strategy_replay: Whether to retain average-policy observations.
            Standalone SD-CFR disables this; advantage replay is unaffected.
        final_policy_network_train_steps: Number of policy-gradient steps for
            the final extraction in final-only mode. Defaults to
            ``policy_network_train_steps``.
        advantage_network_train_steps: SGD steps per training session of each
            advantage network.
        reinitialize_advantage_networks: If True (the Brown et al. recipe),
            re-initialise the advantage network from scratch before each
            training session. If False, training continues from the previous
            iteration's weights.
        compute_exploitability: If True, compute exact NashConv at each policy
            checkpoint. False is much cheaper on larger games where it would
            otherwise dominate wall-clock.
        target_processing: Optional preprocessing applied to sampled advantage
            targets before advantage-network fitting. Supported values are
            ``"none"``, ``"standardize"``, ``"clip"``, and
            ``"standardize_clip"``. Replay buffers always retain raw targets.
        target_clip_value: Symmetric clipping threshold for target-processing
            modes that include clipping.
        target_standardize_epsilon: Minimum standard deviation used when
            standardising a batch of advantage targets.
        advantage_replay_sampling: Sampling scheme for advantage-network
            replay. ``"uniform"`` preserves the baseline reservoir-buffer
            behaviour; ``"priority_abs_adv"`` samples in proportion to sampled
            regret-target magnitude when minibatching.
        replay_buffer_type: Replay storage backend. ``"python"`` keeps the
            original list-of-records reservoir; ``"compact"`` stores fixed-shape
            replay tensors in contiguous NumPy arrays and supplies sampled
            minibatches directly to PyTorch without materialising records.
        average_strategy_weighting: Average-policy supervised loss weighting.
            ``"linear"`` applies the baseline CFR-style iteration weighting;
            ``"uniform"`` gives all sampled policy-memory rows equal weight.
        priority_alpha: Exponent applied to priority replay scores.
        priority_epsilon: Positive floor added to every replay priority.
    """

    def __init__(
        self,
        game,
        policy_network_type: str = "mlp",
        policy_network_layers: Sequence[int] = (256, 256),
        advantage_network_type: str = "mlp",
        advantage_network_layers: Sequence[int] = (128, 128),
        num_iterations: int = 100,
        num_traversals: int = 20,
        learning_rate: float = 1e-4,
        learning_rate_schedule: str = "constant",
        learning_rate_end: Optional[float] = None,
        learning_rate_decay_rate: float = 0.5,
        learning_rate_decay_steps: Optional[int] = None,
        learning_rate_warmup_iterations: int = 0,
        batch_size_advantage: Optional[int] = None,
        batch_size_strategy: Optional[int] = None,
        memory_capacity: int = int(1e6),
        policy_network_train_steps: int = 1,
        policy_network_train_every: int = 1,
        evaluation_interval: Optional[int] = None,
        policy_training_mode: str = "intermittent",
        final_policy_network_train_steps: Optional[int] = None,
        advantage_network_train_steps: int = 1,
        reinitialize_advantage_networks: bool = True,
        compute_exploitability: bool = False,
        target_processing: str = "none",
        target_clip_value: float = 1.0,
        target_standardize_epsilon: float = 1e-6,
        advantage_replay_sampling: str = "uniform",
        replay_buffer_type: str = "python",
        average_strategy_weighting: str = "linear",
        priority_alpha: float = 1.0,
        priority_epsilon: float = 1e-6,
        collect_strategy_replay: bool = True,
    ) -> None:
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            # ``_traverse_game_tree`` does not support simultaneous-move games.
            raise ValueError("Simultaneous games are not supported.")
        if int(policy_network_train_every) < 1:
            raise ValueError("policy_network_train_every must be >= 1")
        if evaluation_interval is None:
            evaluation_interval = policy_network_train_every
        if int(evaluation_interval) < 1:
            raise ValueError("evaluation_interval must be >= 1")
        if policy_training_mode not in {"intermittent", "final_only", "disabled"}:
            raise ValueError(
                "policy_training_mode must be 'intermittent', 'final_only', or 'disabled'"
            )
        if not collect_strategy_replay and policy_training_mode != "disabled":
            raise ValueError("Policy fitting requires collect_strategy_replay=True")
        if final_policy_network_train_steps is None:
            final_policy_network_train_steps = policy_network_train_steps
        if int(final_policy_network_train_steps) < 1:
            raise ValueError("final_policy_network_train_steps must be >= 1")

        self._game = game
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_type = str(policy_network_type).lower()
        self._advantage_network_type = str(advantage_network_type).lower()
        self._policy_network_layers = tuple(int(x) for x in policy_network_layers)
        self._advantage_network_layers = tuple(int(x) for x in advantage_network_layers)
        self._collect_strategy_replay = bool(collect_strategy_replay)
        self._uses_shared_advantage_trunk = (
            self._advantage_network_type == "shared_trunk_player_heads"
        )
        self._policy_network_train_steps = int(policy_network_train_steps)
        self._policy_network_train_every = int(policy_network_train_every)
        self._evaluation_interval = int(evaluation_interval)
        self._policy_training_mode = str(policy_training_mode)
        self._final_policy_network_train_steps = int(final_policy_network_train_steps)
        self._advantage_network_train_steps = int(advantage_network_train_steps)
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = int(num_iterations)
        self._num_traversals = int(num_traversals)
        self._reinitialize_advantage_networks = bool(reinitialize_advantage_networks)
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._initial_learning_rate = float(learning_rate)
        self._learning_rate = float(learning_rate)
        self._learning_rate_schedule = str(learning_rate_schedule).lower()
        valid_lr_schedules = {"constant", "linear_decay", "cosine_decay", "step_decay"}
        if self._learning_rate_schedule not in valid_lr_schedules:
            raise ValueError(
                f"Unknown learning_rate_schedule={learning_rate_schedule!r}. "
                f"Expected one of {sorted(valid_lr_schedules)}."
            )
        self._learning_rate_end = (
            self._initial_learning_rate
            if self._learning_rate_schedule == "constant"
            else (
                float(learning_rate_end)
                if learning_rate_end is not None
                else 0.1 * self._initial_learning_rate
            )
        )
        self._learning_rate_decay_rate = float(learning_rate_decay_rate)
        if self._learning_rate_decay_rate <= 0.0:
            raise ValueError("learning_rate_decay_rate must be positive")
        self._learning_rate_decay_steps = (
            int(learning_rate_decay_steps)
            if learning_rate_decay_steps is not None
            else max(1, self._num_iterations // 3)
        )
        if self._learning_rate_decay_steps < 1:
            raise ValueError("learning_rate_decay_steps must be >= 1")
        self._learning_rate_warmup_iterations = int(learning_rate_warmup_iterations)
        if self._learning_rate_warmup_iterations < 0:
            raise ValueError("learning_rate_warmup_iterations must be >= 0")
        self.learning_rate_history: List[float] = []
        self.checkpoint_learning_rates: List[float] = []
        self._compute_exploitability = bool(compute_exploitability)
        valid_target_processing = {"none", "standardize", "clip", "standardize_clip"}
        self._target_processing = str(target_processing).lower()
        if self._target_processing not in valid_target_processing:
            raise ValueError(
                f"target_processing must be one of {sorted(valid_target_processing)}, "
                f"got {target_processing!r}"
            )
        self._target_clip_value = float(target_clip_value)
        if self._target_clip_value <= 0.0:
            raise ValueError("target_clip_value must be positive")
        self._target_standardize_epsilon = float(target_standardize_epsilon)
        if self._target_standardize_epsilon <= 0.0:
            raise ValueError("target_standardize_epsilon must be positive")
        valid_replay_sampling = {"uniform", "priority_abs_adv"}
        self._advantage_replay_sampling = str(advantage_replay_sampling).lower()
        if self._advantage_replay_sampling not in valid_replay_sampling:
            raise ValueError(
                "advantage_replay_sampling must be one of "
                f"{sorted(valid_replay_sampling)}, got {advantage_replay_sampling!r}"
            )
        valid_average_weighting = {"linear", "uniform"}
        self._average_strategy_weighting = str(average_strategy_weighting).lower()
        if self._average_strategy_weighting not in valid_average_weighting:
            raise ValueError(
                "average_strategy_weighting must be one of "
                f"{sorted(valid_average_weighting)}, got {average_strategy_weighting!r}"
            )
        self._priority_alpha = float(priority_alpha)
        if self._priority_alpha < 0.0:
            raise ValueError("priority_alpha must be non-negative")
        self._priority_epsilon = float(priority_epsilon)
        if self._priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        self._policy_training_events = 0
        self._policy_gradient_steps = 0
        self._iterations_since_policy_train = 0
        self._policy_network_has_been_trained = False
        self._replay_buffer_type = str(replay_buffer_type).lower()

        # Average-policy network.
        self._strategy_memories = make_strategy_buffer(
            memory_capacity if self._collect_strategy_replay else 1,
            self._replay_buffer_type,
            info_state_size=self._embedding_size,
            num_actions=self._num_actions,
        )
        self._policy_network = build_network(
            self._policy_network_type,
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
        )
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=self._learning_rate
        )

        # Per-player advantage networks.
        self._advantage_memories = [
            make_advantage_buffer(
                memory_capacity,
                self._replay_buffer_type,
                info_state_size=self._embedding_size,
                num_actions=self._num_actions,
            )
            for _ in range(self._num_players)
        ]
        if self._uses_shared_advantage_trunk:
            self._advantage_networks = build_shared_trunk_player_heads(
                self._embedding_size,
                list(advantage_network_layers),
                self._num_actions,
                self._num_players,
            )
        else:
            self._advantage_networks = [
                build_network(
                    self._advantage_network_type,
                    self._embedding_size,
                    list(advantage_network_layers),
                    self._num_actions,
                )
                for _ in range(self._num_players)
            ]
        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = self._make_advantage_optimizers()

        # Diagnostics.
        self._nodes_touched = 0
        self._nodes_touched_history: List[int] = []
        self._average_policy_value_history: List[float] = []
        self._cumulative_traversal_collection_seconds = 0.0
        self._last_advantage_grad_norm = [float("nan")] * self._num_players
        self._last_policy_grad_norm = float("nan")
        self._last_processed_advantage_target_mean = [
            float("nan")
        ] * self._num_players
        self._last_processed_advantage_target_variance = [
            float("nan")
        ] * self._num_players
        self._last_processed_advantage_target_abs_mean = [
            float("nan")
        ] * self._num_players
        self._last_target_standardization_mean = [
            float("nan")
        ] * self._num_players
        self._last_target_standardization_scale = [
            float("nan")
        ] * self._num_players
        self._last_target_clip_fraction = [float("nan")] * self._num_players
        self._last_advantage_priority_effective_sample_size = [
            float("nan")
        ] * self._num_players

        # One-shot warning state for under-sized buffer training.
        self._warned_advantage_buffer_too_small = [False] * self._num_players
        self._warned_strategy_buffer_too_small = False

    # ------------------------------------------------------------------ helpers

    @property
    def advantage_buffers(self) -> List[ReservoirBuffer]:
        return self._advantage_memories

    @property
    def strategy_buffer(self) -> ReservoirBuffer:
        return self._strategy_memories

    @property
    def execution_backend(self) -> str:
        return "sequential"

    @property
    def parallel_num_workers(self) -> int:
        return 1

    def clear_advantage_buffers(self) -> None:
        for buf in self._advantage_memories:
            buf.clear()

    def reinitialize_advantage_network(self, player: int) -> None:
        self._advantage_networks[player].reset()
        self._optimizer_advantages = self._make_advantage_optimizers()

    def reinitialize_advantage_networks(self) -> None:
        if self._uses_shared_advantage_trunk:
            self._advantage_networks[0].trunk.reset()
            for net in self._advantage_networks:
                if hasattr(net, "reset_head"):
                    net.reset_head()
                else:
                    net.reset()
        else:
            for net in self._advantage_networks:
                net.reset()
        self._optimizer_advantages = self._make_advantage_optimizers()

    def _unique_advantage_parameters(self) -> List[torch.nn.Parameter]:
        seen = set()
        params = []
        for net in self._advantage_networks:
            for parameter in net.parameters():
                ident = id(parameter)
                if ident not in seen:
                    seen.add(ident)
                    params.append(parameter)
        return params

    def _make_advantage_optimizers(self):
        if self._uses_shared_advantage_trunk:
            shared_optimizer = torch.optim.Adam(
                self._unique_advantage_parameters(), lr=self._learning_rate
            )
            return [shared_optimizer for _ in range(self._num_players)]
        return [
            torch.optim.Adam(net.parameters(), lr=self._learning_rate)
            for net in self._advantage_networks
        ]

    def _learning_rate_at_iteration(self, iteration: int) -> float:
        """Returns the learning rate for a one-indexed CFR iteration."""
        iteration = int(iteration)
        if (
            self._learning_rate_warmup_iterations > 0
            and iteration <= self._learning_rate_warmup_iterations
        ):
            return (
                self._initial_learning_rate
                * iteration
                / float(self._learning_rate_warmup_iterations)
            )

        if self._learning_rate_schedule == "constant":
            return self._initial_learning_rate

        denom = max(1, self._num_iterations - self._learning_rate_warmup_iterations)
        progress = (iteration - self._learning_rate_warmup_iterations) / float(denom)
        progress = min(1.0, max(0.0, progress))

        if self._learning_rate_schedule == "linear_decay":
            return self._initial_learning_rate + progress * (
                self._learning_rate_end - self._initial_learning_rate
            )

        if self._learning_rate_schedule == "cosine_decay":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self._learning_rate_end + (
                self._initial_learning_rate - self._learning_rate_end
            ) * cosine

        if self._learning_rate_schedule == "step_decay":
            steps = max(
                0,
                (iteration - self._learning_rate_warmup_iterations)
                // self._learning_rate_decay_steps,
            )
            lr = self._initial_learning_rate * (self._learning_rate_decay_rate ** steps)
            return max(self._learning_rate_end, lr)

        raise ValueError(f"Unsupported learning-rate schedule: {self._learning_rate_schedule}")

    def _set_learning_rate(self, learning_rate: float) -> None:
        """Updates all neural-network optimisers to the supplied learning rate."""
        self._learning_rate = float(learning_rate)
        for optimizer in [self._optimizer_policy, *self._optimizer_advantages]:
            for group in optimizer.param_groups:
                group["lr"] = self._learning_rate

    @staticmethod
    def _as_scalar_iteration(value) -> float:
        """Returns a scalar float from a stored replay-buffer iteration value."""
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        flat = arr.reshape(-1)
        if flat.size != 1:
            raise ValueError(
                f"Expected scalar iteration value, got shape {arr.shape} "
                f"with type {type(value)}"
            )
        return float(flat[0])

    @staticmethod
    def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
        """Returns the global L2 norm of gradients for a parameter iterable."""
        total_sq = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                total_sq += float(torch.sum(grad * grad).cpu().item())
        return math.sqrt(total_sq)

    # ---------------------------------------------------------------- training

    def solve(
        self,
        post_iteration_callback: Optional[
            Callable[["DeepCFRSolver", int], None]
        ] = None,
        post_player_update_callback: Optional[
            Callable[["DeepCFRSolver", int, int], None]
        ] = None,
        max_training_seconds: Optional[float] = None,
        training_clock: Optional[Callable[[], float]] = None,
    ) -> SolveResult:
        """Runs one fixed-budget Deep CFR training phase.

        ``post_iteration_callback``, when supplied, is called after the policy
        training decision for each completed CFR iteration and receives the
        solver plus its global completed-iteration count. This permits
        lightweight observation or snapshotting without splitting one training
        trajectory into multiple calls to :meth:`solve`.

        ``post_player_update_callback`` runs immediately after each player's
        advantage fit, receiving solver, player and one-indexed iteration.
        SD-CFR uses it to retain the historical strategy at the correct phase.
        An optional positive finite time budget stops after a complete outer
        iteration. By default callbacks count towards the budget. An optional
        monotonic training clock may exclude checkpoint I/O; it must not
        exclude traversal, optimisation or historical-strategy capture.
        """
        if max_training_seconds is not None and (
            not math.isfinite(float(max_training_seconds)) or float(max_training_seconds) <= 0
        ):
            raise ValueError("max_training_seconds must be positive and finite")
        start_time = time.perf_counter()
        budget_clock = training_clock or time.perf_counter
        budget_start = budget_clock()
        advantage_losses: Dict[int, List[float]] = collections.defaultdict(list)
        policy_losses_at_checkpoints: List[Optional[float]] = []
        convs: List[float] = []
        nodes_touched_history: List[int] = []
        average_policy_values: List[float] = []
        diagnostics: Dict[str, list] = collections.defaultdict(list)

        for it in range(self._num_iterations):
            current_lr = self._learning_rate_at_iteration(self._iteration)
            self._set_learning_rate(current_lr)
            self.learning_rate_history.append(float(current_lr))

            # Collect samples and train each advantage network.
            if (
                self._reinitialize_advantage_networks
                and self._uses_shared_advantage_trunk
            ):
                self.reinitialize_advantage_networks()
            collection_seconds_this_iteration = 0.0
            for p in range(self._num_players):
                collection_start = time.perf_counter()
                self._collect_traversals_for_player(p)
                collection_elapsed = time.perf_counter() - collection_start
                collection_seconds_this_iteration += collection_elapsed
                self._cumulative_traversal_collection_seconds += collection_elapsed

                if (
                    self._reinitialize_advantage_networks
                    and not self._uses_shared_advantage_trunk
                ):
                    self.reinitialize_advantage_network(p)

                advantage_losses[p].append(self._learn_advantage_network(p))
                if post_player_update_callback is not None:
                    post_player_update_callback(self, int(p), int(self._iteration))

            # End-of-iteration bookkeeping.
            self._iteration += 1

            # Train the average-policy network either intermittently or once
            # at the end, depending on the experiment.
            if self._policy_training_mode == "disabled":
                train_now = False
            elif self._policy_training_mode == "final_only":
                train_now = it == self._num_iterations - 1
            else:
                train_now = (((it + 1) % self._policy_network_train_every) == 0) or (
                    it == self._num_iterations - 1
                )
            evaluate_now = (((it + 1) % self._evaluation_interval) == 0) or (
                it == self._num_iterations - 1
            )

            pol_loss = None
            if train_now:
                original_policy_steps = self._policy_network_train_steps
                if self._policy_training_mode == "final_only":
                    self._policy_network_train_steps = (
                        self._final_policy_network_train_steps
                    )
                try:
                    pol_loss = self._learn_strategy_network()
                finally:
                    self._policy_network_train_steps = original_policy_steps
                self._policy_training_events += 1
                self._iterations_since_policy_train = 0
                if pol_loss is not None:
                    if self._policy_training_mode == "final_only":
                        self._policy_gradient_steps += (
                            self._final_policy_network_train_steps
                        )
                    else:
                        self._policy_gradient_steps += self._policy_network_train_steps
                    self._policy_network_has_been_trained = True
            else:
                self._iterations_since_policy_train += 1

            if post_iteration_callback is not None:
                post_iteration_callback(self, int(self._iteration - 1))

            time_limit_reached = (
                max_training_seconds is not None
                and budget_clock() - budget_start >= float(max_training_seconds)
            )
            evaluate_now = evaluate_now or time_limit_reached
            if not evaluate_now:
                continue

            policy_losses_at_checkpoints.append(pol_loss)

            nodes_touched_history.append(self._nodes_touched)
            self.checkpoint_learning_rates.append(float(self._learning_rate))

            if self._policy_network_has_been_trained:
                if self._compute_exploitability:
                    tab_pol = policy.tabular_policy_from_callable(
                        self._game, self.action_probabilities
                    )
                    avg_policy_value = expected_game_score.policy_value(
                        self._game.new_initial_state(), [tab_pol] * self._num_players
                    )[0]
                    average_policy_values.append(avg_policy_value)
                    conv = exploitability.nash_conv(self._game, tab_pol)
                    convs.append(conv)
                    policy_diag = self._policy_network_diagnostics()
                else:
                    # Exact tabular evaluation and the full-tree policy
                    # diagnostics are intractable for FHP. Keep checkpoint
                    # metrics explicit as unavailable and evaluate saved
                    # policies later with sampled head-to-head play.
                    average_policy_values.append(float("nan"))
                    convs.append(float("nan"))
                    policy_diag = {
                        "legal_action_mass_mean": float("nan"),
                        "legal_action_mass_min": float("nan"),
                        "entropy_mean": float("nan"),
                        "normalized_entropy_mean": float("nan"),
                    }
            else:
                average_policy_values.append(float("nan"))
                convs.append(float("nan"))
                policy_diag = {
                    "legal_action_mass_mean": float("nan"),
                    "legal_action_mass_min": float("nan"),
                    "entropy_mean": float("nan"),
                    "normalized_entropy_mean": float("nan"),
                }
            target_diag = self._advantage_target_summary()

            diagnostics["iteration"].append(int(it + 1))
            diagnostics["wall_clock_seconds"].append(
                float(time.perf_counter() - start_time)
            )
            diagnostics["traversal_collection_seconds"].append(
                float(collection_seconds_this_iteration)
            )
            diagnostics["cumulative_traversal_collection_seconds"].append(
                float(self._cumulative_traversal_collection_seconds)
            )
            diagnostics["learning_rate"].append(float(self._learning_rate))
            diagnostics["strategy_buffer_size"].append(int(len(self._strategy_memories)))
            diagnostics["advantage_buffer_size_player_0"].append(
                int(len(self._advantage_memories[0]))
            )
            diagnostics["advantage_buffer_size_player_1"].append(
                int(len(self._advantage_memories[1]))
            )
            diagnostics["policy_loss"].append(
                float("nan") if pol_loss is None else float(pol_loss)
            )
            diagnostics["advantage_grad_norm_player_0"].append(
                float(self._last_advantage_grad_norm[0])
            )
            diagnostics["advantage_grad_norm_player_1"].append(
                float(self._last_advantage_grad_norm[1])
            )
            diagnostics["policy_grad_norm"].append(float(self._last_policy_grad_norm))
            diagnostics["policy_training_events"].append(
                int(self._policy_training_events)
            )
            diagnostics["policy_gradient_steps"].append(
                int(self._policy_gradient_steps)
            )
            diagnostics["iterations_since_policy_train"].append(
                int(self._iterations_since_policy_train)
            )
            diagnostics["policy_network_has_been_trained"].append(
                bool(self._policy_network_has_been_trained)
            )
            diagnostics["trained_policy_this_iteration"].append(bool(train_now))
            diagnostics["legal_action_mass_mean"].append(
                float(policy_diag["legal_action_mass_mean"])
            )
            diagnostics["legal_action_mass_min"].append(
                float(policy_diag["legal_action_mass_min"])
            )
            diagnostics["policy_entropy_mean"].append(float(policy_diag["entropy_mean"]))
            diagnostics["policy_normalized_entropy_mean"].append(
                float(policy_diag["normalized_entropy_mean"])
            )
            diagnostics["advantage_target_count"].append(int(target_diag["count"]))
            diagnostics["advantage_target_mean"].append(float(target_diag["mean"]))
            diagnostics["advantage_target_variance"].append(
                float(target_diag["variance"])
            )
            diagnostics["advantage_target_abs_mean"].append(
                float(target_diag["abs_mean"])
            )
            diagnostics["processed_advantage_target_mean_player_0"].append(
                float(self._last_processed_advantage_target_mean[0])
            )
            diagnostics["processed_advantage_target_mean_player_1"].append(
                float(self._last_processed_advantage_target_mean[1])
            )
            diagnostics["processed_advantage_target_variance_player_0"].append(
                float(self._last_processed_advantage_target_variance[0])
            )
            diagnostics["processed_advantage_target_variance_player_1"].append(
                float(self._last_processed_advantage_target_variance[1])
            )
            diagnostics["processed_advantage_target_abs_mean_player_0"].append(
                float(self._last_processed_advantage_target_abs_mean[0])
            )
            diagnostics["processed_advantage_target_abs_mean_player_1"].append(
                float(self._last_processed_advantage_target_abs_mean[1])
            )
            diagnostics["target_standardization_mean_player_0"].append(
                float(self._last_target_standardization_mean[0])
            )
            diagnostics["target_standardization_mean_player_1"].append(
                float(self._last_target_standardization_mean[1])
            )
            diagnostics["target_standardization_scale_player_0"].append(
                float(self._last_target_standardization_scale[0])
            )
            diagnostics["target_standardization_scale_player_1"].append(
                float(self._last_target_standardization_scale[1])
            )
            diagnostics["target_clip_fraction_player_0"].append(
                float(self._last_target_clip_fraction[0])
            )
            diagnostics["target_clip_fraction_player_1"].append(
                float(self._last_target_clip_fraction[1])
            )
            diagnostics["advantage_priority_effective_sample_size_player_0"].append(
                float(self._last_advantage_priority_effective_sample_size[0])
            )
            diagnostics["advantage_priority_effective_sample_size_player_1"].append(
                float(self._last_advantage_priority_effective_sample_size[1])
            )
            diagnostics["advantage_priority_effective_sample_size"].append(
                float(
                    np.nanmean(
                        np.asarray(
                            self._last_advantage_priority_effective_sample_size,
                            dtype=np.float64,
                        )
                    )
                )
            )

            if time_limit_reached:
                break

        self._nodes_touched_history = nodes_touched_history
        self._average_policy_value_history = average_policy_values

        return SolveResult(
            policy_network=self._policy_network,
            advantage_losses=dict(advantage_losses),
            policy_losses=policy_losses_at_checkpoints,
            nash_conv=convs,
            nodes_touched=nodes_touched_history,
            average_policy_value=average_policy_values,
            diagnostics=dict(diagnostics),
        )

    # ----------------------------------------------------------- traversal

    def _collect_traversals_for_player(self, player: int) -> None:
        """Collects this iteration's traversal samples for one player."""
        with self._advantage_inference_mode():
            for _ in range(self._num_traversals):
                self._traverse_game_tree(self._root_node, int(player))

    @contextlib.contextmanager
    def _advantage_inference_mode(self):
        """Sets advantage networks to eval once for a collection phase."""
        training_modes = [network.training for network in self._advantage_networks]
        for network in self._advantage_networks:
            network.eval()
        try:
            with torch.inference_mode():
                yield
        finally:
            for network, was_training in zip(
                self._advantage_networks, training_modes
            ):
                network.train(was_training)

    def _traverse_game_tree(self, state, player: int) -> float:
        """External-sampling traversal that populates the replay buffers.

        Returns the expected payoff for ``player`` from ``state``.
        """
        self._nodes_touched += 1
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = int(np.random.choice(chance_outcome, p=chance_proba))
            return self._traverse_game_tree(state.child(action), player)

        if state.current_player() == player:
            info_state = np.asarray(
                state.information_state_tensor(player), dtype=np.float32
            )
            _, strategy = self._sample_action_from_advantage(
                state, player, info_state=info_state
            )
            legal_actions = state.legal_actions()
            expected_payoff: Dict[int, float] = {}
            for action in legal_actions:
                expected_payoff[action] = self._traverse_game_tree(
                    state.child(action), player
                )

            cfv = 0.0
            for action in legal_actions:
                cfv += float(strategy[action]) * float(expected_payoff[action])

            sampled_regret = np.zeros(self._num_actions, dtype=np.float32)
            for action in legal_actions:
                sampled_regret[action] = float(expected_payoff[action]) - cfv

            self._advantage_memories[player].add(
                AdvantageMemory(
                    info_state,
                    int(self._iteration),
                    sampled_regret,
                )
            )
            return cfv

        # Non-traversing player: store their regret-matched policy as an
        # average-policy training target and sample one action to descend.
        other_player = state.current_player()
        info_state = np.asarray(
            state.information_state_tensor(other_player), dtype=np.float32
        )
        _, strategy = self._sample_action_from_advantage(
            state, other_player, info_state=info_state
        )
        probs = np.asarray(strategy, dtype=np.float64)
        total = probs.sum()
        if total <= 0.0:
            # Fallback: uniform over legal actions. Shouldn't happen because
            # ``_sample_action_from_advantage`` already falls back to uniform,
            # but guard defensively.
            legal_actions = state.legal_actions(other_player)
            probs = np.zeros(self._num_actions, dtype=np.float64)
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        else:
            probs = probs / total
        sampled_action = int(np.random.choice(self._num_actions, p=probs))
        if self._collect_strategy_replay:
            self._strategy_memories.add(
                StrategyMemory(
                    info_state,
                    int(self._iteration),
                    strategy,
                )
            )
        return self._traverse_game_tree(state.child(sampled_action), player)

    def _sample_action_from_advantage(
        self,
        state,
        player: int,
        *,
        info_state: Optional[np.ndarray] = None,
    ) -> Tuple[List[float], np.ndarray]:
        """Returns the advantages and the regret-matched policy for ``player``.

        The matched policy has zero mass on illegal actions and sums to 1 over
        legal actions. When all positive regrets are zero, the policy falls
        back to uniform over legal actions (avoiding deterministic tie-breaking
        that would suppress exploration).
        """
        if info_state is None:
            info_state = np.asarray(
                state.information_state_tensor(player), dtype=np.float32
            )
        legal_actions = state.legal_actions(player)
        net = self._advantage_networks[player]
        state_tensor = torch.from_numpy(
            np.asarray(info_state, dtype=np.float32).reshape(1, -1)
        )
        if torch.is_inference_mode_enabled():
            raw_advantages = net(state_tensor)[0].cpu().numpy()
        else:
            was_training = net.training
            net.eval()
            with torch.inference_mode():
                try:
                    raw_advantages = net(state_tensor)[0].cpu().numpy()
                finally:
                    net.train(was_training)
        advantages = [max(0.0, a) for a in raw_advantages]
        cumulative_regret = float(sum(advantages[a] for a in legal_actions))
        matched_regrets = np.zeros(self._num_actions, dtype=np.float32)
        if cumulative_regret > 0.0:
            for action in legal_actions:
                matched_regrets[action] = advantages[action] / cumulative_regret
        else:
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                matched_regrets[action] = uniform
        return advantages, matched_regrets

    def action_probabilities(self, state, player_id=None) -> Dict[int, float]:
        """Returns legal-action probabilities for ``state`` under the avg policy.

        Illegal actions are masked out and the remaining mass is renormalised.
        OpenSpiel's tabular policy and exploitability utilities require the
        returned distribution to sum to 1 over legal actions.
        """
        cur_player = state.current_player() if player_id is None else player_id
        legal_actions = state.legal_actions(cur_player)
        if not legal_actions:
            return {}
        info_state_vector = np.asarray(
            state.information_state_tensor(cur_player), dtype=np.float32
        )
        if info_state_vector.ndim == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        was_training = self._policy_network.training
        self._policy_network.eval()
        with torch.no_grad():
            try:
                logits = self._policy_network(
                    torch.as_tensor(info_state_vector, dtype=torch.float32)
                )[0]
                legal_logits = logits[legal_actions]
                legal_probs = self._policy_sm(legal_logits).cpu().numpy()
            finally:
                if was_training:
                    self._policy_network.train()
        return {
            int(action): float(prob)
            for action, prob in zip(legal_actions, legal_probs)
        }

    # ---------------------------------------------------------- diagnostics

    def _policy_network_diagnostics(self) -> Dict[str, float]:
        """Summarises legal-action mass and entropy over reachable infosets.

        Walks the game tree once with an iterative DFS and a ``set`` of seen
        infosets keyed by the OpenSpiel ``information_state_string``. This
        avoids the previous tuple-of-floats key (which is expensive on larger
        games) and the recursion depth limit.
        """
        legal_masses: List[float] = []
        entropies: List[float] = []
        normalised_entropies: List[float] = []
        seen: set = set()

        stack = [self._game.new_initial_state()]
        while stack:
            state = stack.pop()
            if state.is_terminal():
                continue
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    stack.append(state.child(action))
                continue

            player = state.current_player()
            legal_actions = state.legal_actions(player)
            if not legal_actions:
                continue

            key = (player, state.information_state_string(player))
            if key not in seen:
                seen.add(key)
                info_state = np.asarray(
                    state.information_state_tensor(player), dtype=np.float32
                )
                was_training = self._policy_network.training
                self._policy_network.eval()
                with torch.no_grad():
                    try:
                        logits = self._policy_network(
                            torch.as_tensor(
                                np.expand_dims(info_state, axis=0), dtype=torch.float32
                            )
                        )[0]
                        full_probs = self._policy_sm(logits).cpu().numpy()
                    finally:
                        if was_training:
                            self._policy_network.train()
                legal_mass = float(np.sum(full_probs[legal_actions]))
                if legal_mass > 0:
                    legal_probs = full_probs[legal_actions] / legal_mass
                else:
                    legal_probs = (
                        np.ones(len(legal_actions), dtype=np.float32)
                        / len(legal_actions)
                    )
                entropy = float(-np.sum(legal_probs * np.log(legal_probs + 1e-12)))
                if len(legal_actions) > 1:
                    max_entropy = float(np.log(len(legal_actions)))
                    norm_entropy = entropy / max_entropy
                else:
                    norm_entropy = 0.0
                legal_masses.append(legal_mass)
                entropies.append(entropy)
                normalised_entropies.append(norm_entropy)

            for action in legal_actions:
                stack.append(state.child(action))

        if not legal_masses:
            return {
                "legal_action_mass_mean": float("nan"),
                "legal_action_mass_min": float("nan"),
                "entropy_mean": float("nan"),
                "normalized_entropy_mean": float("nan"),
            }

        return {
            "legal_action_mass_mean": float(np.mean(legal_masses)),
            "legal_action_mass_min": float(np.min(legal_masses)),
            "entropy_mean": float(np.mean(entropies)),
            "normalized_entropy_mean": float(np.mean(normalised_entropies)),
        }

    def _advantage_target_summary(self) -> Dict[str, float]:
        """Summarises sampled advantage targets, optionally subsampled.

        When buffers are large we take a uniform random subsample of at most
        ``_ADVANTAGE_DIAGNOSTIC_MAX_SAMPLES`` advantage vectors so that the
        diagnostic cost stays bounded. Welford-style streaming statistics could
        also be used, but the closed-form numpy reduction over a fixed-size
        sample is simpler and good enough for thesis diagnostics.
        """
        total = sum(len(buf) for buf in self._advantage_memories)
        if total == 0:
            return {"count": 0, "mean": float("nan"), "variance": float("nan"),
                    "abs_mean": float("nan")}

        if total <= _ADVANTAGE_DIAGNOSTIC_MAX_SAMPLES:
            chunks = []
            for buf in self._advantage_memories:
                as_batch = getattr(buf, "as_batch", None)
                if callable(as_batch):
                    chunks.append(as_batch()["targets"].reshape(-1))
                else:
                    for sample in buf:
                        chunks.append(
                            np.asarray(sample.advantage, dtype=np.float32).ravel()
                        )
        else:
            # Sample without replacement across both buffers proportional to
            # their sizes.
            chunks = []
            sizes = [len(buf) for buf in self._advantage_memories]
            total_size = sum(sizes)
            for size, buf in zip(sizes, self._advantage_memories):
                share = max(
                    1, int(round(_ADVANTAGE_DIAGNOSTIC_MAX_SAMPLES * size / total_size))
                )
                share = min(share, size)
                sample_batch = getattr(buf, "sample_batch", None)
                if callable(sample_batch):
                    if share == size:
                        chunks.append(buf.as_batch()["targets"].reshape(-1))
                    else:
                        chunks.append(
                            sample_batch(share)["targets"].reshape(-1)
                        )
                else:
                    if share == size:
                        samples = list(buf)
                    else:
                        samples = random.sample(list(buf), share)
                    for sample in samples:
                        chunks.append(
                            np.asarray(sample.advantage, dtype=np.float32).ravel()
                        )

        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        if values.size == 0:
            return {"count": 0, "mean": float("nan"), "variance": float("nan"),
                    "abs_mean": float("nan")}
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),
            "abs_mean": float(np.mean(np.abs(values))),
        }

    def _process_advantage_targets(
        self, targets: np.ndarray, player: int
    ) -> np.ndarray:
        """Transforms advantage targets for the supervised fitting loss.

        Replay buffers keep raw sampled regrets. This method only changes the
        batch passed to the advantage-network loss and records diagnostics for
        the current player/training step.
        """
        raw = np.asarray(targets, dtype=np.float32)
        processed = raw.copy()
        raw_mean = float(np.mean(raw)) if raw.size else float("nan")

        standardization_mean = 0.0
        standardization_scale = 1.0
        if self._target_processing in {"standardize", "standardize_clip"}:
            standardization_mean = raw_mean
            standardization_scale = max(
                float(np.std(processed)), self._target_standardize_epsilon
            )
            processed = (processed - standardization_mean) / standardization_scale

        clip_fraction = 0.0
        if self._target_processing in {"clip", "standardize_clip"}:
            before_clip = processed.copy()
            processed = np.clip(
                processed,
                -self._target_clip_value,
                self._target_clip_value,
            )
            clip_fraction = (
                float(np.mean(before_clip != processed)) if processed.size else 0.0
            )

        self._last_processed_advantage_target_mean[player] = (
            float(np.mean(processed)) if processed.size else float("nan")
        )
        self._last_processed_advantage_target_variance[player] = (
            float(np.var(processed)) if processed.size else float("nan")
        )
        self._last_processed_advantage_target_abs_mean[player] = (
            float(np.mean(np.abs(processed))) if processed.size else float("nan")
        )
        self._last_target_standardization_mean[player] = float(standardization_mean)
        self._last_target_standardization_scale[player] = float(standardization_scale)
        self._last_target_clip_fraction[player] = float(clip_fraction)
        return processed.astype(np.float32)

    # ------------------------------------------------------------- learners

    def _learn_advantage_network(self, player: int) -> Optional[float]:
        """One training session for the advantage network of ``player``."""
        last_loss: Optional[float] = None
        grad_norms: List[float] = []
        buffer = self._advantage_memories[player]
        self._advantage_networks[player].train()

        for _ in range(self._advantage_network_train_steps):
            batch = self._draw_advantage_batch(player, buffer)
            if batch is None:
                return None

            info_states = batch["info_states"]
            advantages = batch["targets"]
            advantages = self._process_advantage_targets(advantages, player)
            iterations = batch["iterations"].astype(np.float32, copy=False)

            self._optimizer_advantages[player].zero_grad()
            iters = torch.from_numpy(
                np.sqrt(iterations).reshape(-1, 1)
            )
            outputs = self._advantage_networks[player](
                torch.from_numpy(info_states)
            )
            targets = torch.from_numpy(advantages)
            loss = self._loss_advantages(iters * outputs, iters * targets)
            loss.backward()
            grad_norms.append(
                self._gradient_norm(self._advantage_networks[player].parameters())
            )
            self._optimizer_advantages[player].step()
            last_loss = float(loss.detach().cpu().item())

        self._last_advantage_grad_norm[player] = (
            float(np.mean(grad_norms)) if grad_norms else float("nan")
        )
        return last_loss

    def _draw_advantage_batch(
        self, player: int, buffer: ReservoirBuffer
    ) -> Optional[ReplayBatch]:
        """Returns typed learner arrays, using zero-object compact sampling."""
        if len(buffer) == 0:
            self._last_advantage_priority_effective_sample_size[player] = float(
                "nan"
            )
            return None

        sample_batch = getattr(buffer, "sample_batch", None)
        if not callable(sample_batch):
            samples = self._draw_advantage_samples(player, buffer)
            if not samples:
                return None
            return records_to_batch(
                samples,
                info_state_size=self._embedding_size,
                target_size=self._num_actions,
                target_attr="advantage",
            )

        if self._advantage_replay_sampling == "uniform":
            self._last_advantage_priority_effective_sample_size[player] = float(
                len(buffer)
            )

        batch_size = self._batch_size_advantage
        if not batch_size or batch_size > len(buffer):
            if batch_size and not self._warned_advantage_buffer_too_small[player]:
                _LOGGER.warning(
                    "Advantage buffer for player %d has %d samples but "
                    "batch_size_advantage=%d. Falling back to training on "
                    "the full buffer for this iteration. Subsequent "
                    "occurrences are silenced.",
                    player,
                    len(buffer),
                    batch_size,
                )
                self._warned_advantage_buffer_too_small[player] = True
            if self._advantage_replay_sampling == "priority_abs_adv":
                self._record_compact_priority_effective_sample_size(player, buffer)
            return buffer.as_batch()

        if self._advantage_replay_sampling == "priority_abs_adv":
            probabilities = self._compact_priority_probabilities(buffer)
            self._last_advantage_priority_effective_sample_size[player] = float(
                1.0 / np.sum(probabilities * probabilities)
            )
            return sample_batch(batch_size, probabilities=probabilities)
        return sample_batch(batch_size)

    def _compact_priority_probabilities(self, buffer) -> np.ndarray:
        raw_scores = buffer.target_abs_mean()
        priorities = np.power(
            raw_scores + self._priority_epsilon, self._priority_alpha
        )
        total = float(np.sum(priorities))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(buffer), 1.0 / len(buffer), dtype=np.float64)
        return priorities / total

    def _record_compact_priority_effective_sample_size(
        self, player: int, buffer
    ) -> None:
        probabilities = self._compact_priority_probabilities(buffer)
        self._last_advantage_priority_effective_sample_size[player] = float(
            1.0 / np.sum(probabilities * probabilities)
        )

    def _draw_advantage_samples(
        self, player: int, buffer: ReservoirBuffer
    ) -> List[AdvantageMemory]:
        """Returns a batch of advantage samples or [] if the buffer is empty."""
        if len(buffer) == 0:
            self._last_advantage_priority_effective_sample_size[player] = float("nan")
            return []
        if self._advantage_replay_sampling == "uniform":
            self._last_advantage_priority_effective_sample_size[player] = float(
                len(buffer)
            )
        if self._batch_size_advantage:
            batch_size = self._batch_size_advantage
            if batch_size > len(buffer):
                if not self._warned_advantage_buffer_too_small[player]:
                    _LOGGER.warning(
                        "Advantage buffer for player %d has %d samples but "
                        "batch_size_advantage=%d. Falling back to training on "
                        "the full buffer for this iteration. Subsequent "
                        "occurrences are silenced.",
                        player,
                        len(buffer),
                        batch_size,
                    )
                    self._warned_advantage_buffer_too_small[player] = True
                self._record_priority_effective_sample_size(player, list(buffer))
                return list(buffer)
            if self._advantage_replay_sampling == "priority_abs_adv":
                return self._sample_priority_advantage_batch(player, buffer, batch_size)
            return buffer.sample(batch_size)
        if self._advantage_replay_sampling == "priority_abs_adv":
            self._record_priority_effective_sample_size(player, list(buffer))
        return list(buffer)

    def _priority_probabilities(
        self, samples: List[AdvantageMemory]
    ) -> np.ndarray:
        """Returns priority replay probabilities for advantage-memory samples."""
        raw_scores = np.asarray(
            [
                float(np.mean(np.abs(np.asarray(sample.advantage, dtype=np.float32))))
                for sample in samples
            ],
            dtype=np.float64,
        )
        priorities = np.power(raw_scores + self._priority_epsilon, self._priority_alpha)
        total = float(np.sum(priorities))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(samples), 1.0 / len(samples), dtype=np.float64)
        return priorities / total

    def _record_priority_effective_sample_size(
        self, player: int, samples: List[AdvantageMemory]
    ) -> None:
        if not samples:
            self._last_advantage_priority_effective_sample_size[player] = float("nan")
            return
        if self._advantage_replay_sampling == "uniform":
            self._last_advantage_priority_effective_sample_size[player] = float(
                len(samples)
            )
            return
        probs = self._priority_probabilities(samples)
        self._last_advantage_priority_effective_sample_size[player] = float(
            1.0 / np.sum(probs * probs)
        )

    def _sample_priority_advantage_batch(
        self, player: int, buffer: ReservoirBuffer, batch_size: int
    ) -> List[AdvantageMemory]:
        if hasattr(buffer, "target_abs_mean") and hasattr(
            buffer, "sample_with_probabilities"
        ):
            raw_scores = buffer.target_abs_mean()
            priorities = np.power(
                raw_scores + self._priority_epsilon, self._priority_alpha
            )
            total = float(np.sum(priorities))
            if not np.isfinite(total) or total <= 0.0:
                probs = np.full(len(buffer), 1.0 / len(buffer), dtype=np.float64)
            else:
                probs = priorities / total
            self._last_advantage_priority_effective_sample_size[player] = float(
                1.0 / np.sum(probs * probs)
            )
            return buffer.sample_with_probabilities(probs, batch_size)

        samples = list(buffer)
        probs = self._priority_probabilities(samples)
        self._last_advantage_priority_effective_sample_size[player] = float(
            1.0 / np.sum(probs * probs)
        )
        indices = np.random.choice(
            len(samples),
            size=int(batch_size),
            replace=False,
            p=probs,
        )
        return [samples[int(i)] for i in indices]

    def _learn_strategy_network(self) -> Optional[float]:
        """One training session for the average-policy network."""
        last_loss: Optional[float] = None
        grad_norms: List[float] = []
        self._policy_network.train()

        for _ in range(self._policy_network_train_steps):
            batch = self._draw_strategy_batch()
            if batch is None:
                return None

            info_states = batch["info_states"]
            action_probs = batch["targets"]
            iterations = batch["iterations"].astype(np.float32, copy=False)

            self._optimizer_policy.zero_grad()
            if self._average_strategy_weighting == "linear":
                sample_weights = np.sqrt(iterations).reshape(-1, 1)
            else:
                sample_weights = np.ones((len(iterations), 1), dtype=np.float32)
            iters = torch.from_numpy(sample_weights)
            ac_probs = torch.from_numpy(action_probs)
            logits = self._policy_network(
                torch.from_numpy(info_states)
            )
            outputs = self._policy_sm(logits)
            loss = self._loss_policy(iters * outputs, iters * ac_probs)
            loss.backward()
            grad_norms.append(self._gradient_norm(self._policy_network.parameters()))
            self._optimizer_policy.step()
            last_loss = float(loss.detach().cpu().item())

        self._last_policy_grad_norm = (
            float(np.mean(grad_norms)) if grad_norms else float("nan")
        )
        return last_loss

    def _draw_strategy_batch(self) -> Optional[ReplayBatch]:
        """Returns typed strategy learner arrays when compact replay is used."""
        buffer = self._strategy_memories
        if len(buffer) == 0:
            return None
        sample_batch = getattr(buffer, "sample_batch", None)
        if not callable(sample_batch):
            samples = self._draw_strategy_samples()
            if not samples:
                return None
            return records_to_batch(
                samples,
                info_state_size=self._embedding_size,
                target_size=self._num_actions,
                target_attr="strategy_action_probs",
            )

        batch_size = self._batch_size_strategy
        if not batch_size or batch_size > len(buffer):
            if batch_size and not self._warned_strategy_buffer_too_small:
                _LOGGER.warning(
                    "Strategy buffer has %d samples but "
                    "batch_size_strategy=%d. Falling back to training on "
                    "the full buffer for this iteration. Subsequent "
                    "occurrences are silenced.",
                    len(buffer),
                    batch_size,
                )
                self._warned_strategy_buffer_too_small = True
            return buffer.as_batch()
        return sample_batch(batch_size)

    def _draw_strategy_samples(self) -> List[StrategyMemory]:
        """Returns a batch of strategy samples or [] if the buffer is empty."""
        buffer = self._strategy_memories
        if len(buffer) == 0:
            return []
        if self._batch_size_strategy:
            batch_size = self._batch_size_strategy
            if batch_size > len(buffer):
                if not self._warned_strategy_buffer_too_small:
                    _LOGGER.warning(
                        "Strategy buffer has %d samples but "
                        "batch_size_strategy=%d. Falling back to training on "
                        "the full buffer for this iteration. Subsequent "
                        "occurrences are silenced.",
                        len(buffer),
                        batch_size,
                    )
                    self._warned_strategy_buffer_too_small = True
                return list(buffer)
            return buffer.sample(batch_size)
        return list(buffer)

    # ---------------------------------------------------------- checkpoints

    def save_full_model(
        self,
        path,
        *,
        include_buffers: bool = True,
        include_rng_state: bool = True,
    ) -> "os.PathLike":
        """Convenience wrapper that ``torch.save``\\ s :meth:`extract_full_model`."""
        from pathlib import Path as _Path

        out_path = _Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.extract_full_model(
                include_buffers=include_buffers,
                include_rng_state=include_rng_state,
            ),
            out_path,
        )
        return out_path

    def save_policy_snapshot(
        self,
        path,
        *,
        seed: int,
        target_iteration: int,
        stage_label: str = "",
        experiment_name: str = "",
        game_name: str = "FHP",
        solver_config=None,
    ) -> "os.PathLike":
        """Convenience wrapper for :func:`deep_cfr_poker.snapshots.save_policy_snapshot`."""
        from .snapshots import save_policy_snapshot as _save

        return _save(
            self,
            path,
            seed=seed,
            target_iteration=target_iteration,
            stage_label=stage_label,
            experiment_name=experiment_name,
            game_name=game_name,
            solver_config=solver_config,
        )

    def extract_full_model(
        self, include_buffers: bool = True, include_rng_state: bool = True
    ) -> dict:
        """Returns a dict suitable for ``torch.save`` and :meth:`load_full_model`."""
        ckpt = {
            "version": 2,
            "iteration": int(self._iteration),
            "policy_state_dict": self._policy_network.state_dict(),
            "policy_opt_state_dict": self._optimizer_policy.state_dict(),
            "advantage_state_dicts": [
                net.state_dict() for net in self._advantage_networks
            ],
            "advantage_opt_state_dicts": [
                opt.state_dict() for opt in self._optimizer_advantages
            ],
            "meta": {
                "num_players": int(self._num_players),
                "num_actions": int(self._num_actions),
                "embedding_size": int(self._embedding_size),
                "policy_network_type": str(self._policy_network_type),
                "advantage_network_type": str(self._advantage_network_type),
                "replay_buffer_type": str(self._replay_buffer_type),
            },
            "training_state": {
                "nodes_touched": int(self._nodes_touched),
                "policy_training_events": int(self._policy_training_events),
                "policy_gradient_steps": int(self._policy_gradient_steps),
                "iterations_since_policy_train": int(
                    self._iterations_since_policy_train
                ),
                "policy_network_has_been_trained": bool(
                    self._policy_network_has_been_trained
                ),
                "learning_rate": float(self._learning_rate),
                "learning_rate_history": list(self.learning_rate_history),
                "checkpoint_learning_rates": list(self.checkpoint_learning_rates),
                "last_processed_advantage_target_mean": list(
                    self._last_processed_advantage_target_mean
                ),
                "last_processed_advantage_target_variance": list(
                    self._last_processed_advantage_target_variance
                ),
                "last_processed_advantage_target_abs_mean": list(
                    self._last_processed_advantage_target_abs_mean
                ),
                "last_target_standardization_mean": list(
                    self._last_target_standardization_mean
                ),
                "last_target_standardization_scale": list(
                    self._last_target_standardization_scale
                ),
                "last_target_clip_fraction": list(self._last_target_clip_fraction),
                "last_advantage_priority_effective_sample_size": list(
                    self._last_advantage_priority_effective_sample_size
                ),
            },
        }

        if include_buffers:
            ckpt["strategy_buffer"] = self._strategy_memories.state_dict()
            ckpt["advantage_buffers"] = [
                buf.state_dict() for buf in self._advantage_memories
            ]

        if include_rng_state:
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                try:
                    rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
                except RuntimeError as exc:
                    _LOGGER.warning(
                        "Failed to capture CUDA RNG state: %s", exc
                    )
            ckpt["rng_state"] = rng_state

        return ckpt

    def load_full_model(
        self,
        ckpt_or_path,
        map_location=None,
        strict: bool = True,
        restore_buffers: bool = True,
        restore_rng_state: bool = True,
    ) -> "DeepCFRSolver":
        """Restores a checkpoint produced by :meth:`extract_full_model`."""

        def _move_optimizer_state_to_device(optimizer, device):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        if isinstance(ckpt_or_path, (str, os.PathLike)):
            ckpt = torch.load(
                ckpt_or_path, map_location=map_location, weights_only=False
            )
        else:
            ckpt = ckpt_or_path

        meta = ckpt.get("meta", {})
        for key, expected in (
            ("num_players", self._num_players),
            ("num_actions", self._num_actions),
            ("embedding_size", self._embedding_size),
        ):
            if key in meta and int(meta[key]) != int(expected):
                raise ValueError(
                    f"Checkpoint metadata mismatch on '{key}': checkpoint says "
                    f"{meta[key]}, current solver has {expected}. The "
                    f"checkpoint was likely produced for a different game."
                )

        self._iteration = int(ckpt.get("iteration", 0))
        training_state = ckpt.get("training_state", {})
        self._nodes_touched = int(training_state.get("nodes_touched", 0))
        self._policy_training_events = int(
            training_state.get("policy_training_events", 0)
        )
        self._policy_gradient_steps = int(
            training_state.get("policy_gradient_steps", 0)
        )
        self._iterations_since_policy_train = int(
            training_state.get("iterations_since_policy_train", 0)
        )
        self._policy_network_has_been_trained = bool(
            training_state.get("policy_network_has_been_trained", False)
        )
        self._learning_rate = float(
            training_state.get("learning_rate", self._learning_rate)
        )
        self.learning_rate_history = list(
            training_state.get("learning_rate_history", [])
        )
        self.checkpoint_learning_rates = list(
            training_state.get("checkpoint_learning_rates", [])
        )
        self._last_processed_advantage_target_mean = list(
            training_state.get(
                "last_processed_advantage_target_mean",
                self._last_processed_advantage_target_mean,
            )
        )
        self._last_processed_advantage_target_variance = list(
            training_state.get(
                "last_processed_advantage_target_variance",
                self._last_processed_advantage_target_variance,
            )
        )
        self._last_processed_advantage_target_abs_mean = list(
            training_state.get(
                "last_processed_advantage_target_abs_mean",
                self._last_processed_advantage_target_abs_mean,
            )
        )
        self._last_target_standardization_mean = list(
            training_state.get(
                "last_target_standardization_mean",
                self._last_target_standardization_mean,
            )
        )
        self._last_target_standardization_scale = list(
            training_state.get(
                "last_target_standardization_scale",
                self._last_target_standardization_scale,
            )
        )
        self._last_target_clip_fraction = list(
            training_state.get(
                "last_target_clip_fraction",
                self._last_target_clip_fraction,
            )
        )
        self._last_advantage_priority_effective_sample_size = list(
            training_state.get(
                "last_advantage_priority_effective_sample_size",
                self._last_advantage_priority_effective_sample_size,
            )
        )

        self._policy_network.load_state_dict(ckpt["policy_state_dict"], strict=strict)
        self._optimizer_policy.load_state_dict(ckpt["policy_opt_state_dict"])
        _move_optimizer_state_to_device(
            self._optimizer_policy, next(self._policy_network.parameters()).device
        )

        for net, sd in zip(self._advantage_networks, ckpt["advantage_state_dicts"]):
            net.load_state_dict(sd, strict=strict)
        for opt, net, sd in zip(
            self._optimizer_advantages,
            self._advantage_networks,
            ckpt["advantage_opt_state_dicts"],
        ):
            opt.load_state_dict(sd)
            _move_optimizer_state_to_device(opt, next(net.parameters()).device)

        self._set_learning_rate(self._learning_rate)

        if restore_buffers:
            if "strategy_buffer" in ckpt:
                self._strategy_memories.load_state_dict(ckpt["strategy_buffer"])
            if "advantage_buffers" in ckpt:
                for buf, saved in zip(self._advantage_memories, ckpt["advantage_buffers"]):
                    buf.load_state_dict(saved)

        if restore_rng_state:
            rng = ckpt.get("rng_state", {})
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and "torch_cuda" in rng:
                try:
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                except RuntimeError as exc:
                    _LOGGER.warning("Failed to restore CUDA RNG state: %s", exc)

        return self
