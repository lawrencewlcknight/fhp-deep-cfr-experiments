"""Standalone SD-CFR using the selected uniform small-game configuration.

This is a reusable algorithm, not a numbered/paid experiment. The existing
archived conventional Deep CFR experiment remains unchanged apart from naming.
"""

from types import MappingProxyType

from .game import load_fhp_game
from .sd_cfr import SDCFRArchive, SampledSDCFRPolicy
from .solver import DeepCFRSolver


SELECTED_SD_CFR_KWARGS = MappingProxyType({
    "num_traversals": 320,
    "advantage_network_layers": (32,) * 8,
    "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
    "learning_rate": 0.004,
    "learning_rate_schedule": "constant",
    "batch_size_advantage": 2048,
    "memory_capacity": 5_000_000,
    "reinitialize_advantage_networks": False,
    "advantage_network_train_steps": 200,
    "target_processing": "standardize",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "advantage_replay_sampling": "uniform",
    "priority_alpha": 1.0,
    "priority_epsilon": 1e-6,
    "average_strategy_weighting": "uniform",
    "replay_buffer_type": "compact",
    "compute_exploitability": False,
    "policy_training_mode": "disabled",
    "collect_strategy_replay": False,
    # Kept only for constructor/RNG compatibility with the source solver.
    # This small network is never fitted or deployed.
    "policy_network_type": "mlp",
    "policy_network_layers": (32, 32),
    "batch_size_strategy": 1024,
    "policy_network_train_steps": 200,
    "policy_network_train_every": 10,
    "evaluation_interval": 25,
})


class SingleDeepCFRSolver(DeepCFRSolver):
    """Train advantages and retain all iteration networks for SD-CFR play.

    Defaults are the selected uniform Experiment 28/29 algorithm: continuous
    warm start, including persistent Adam state. FHP's compact float32 replay
    is retained. Override training size explicitly for smoke tests. No network
    thinning, resampling of archives, or lossy compression is performed.
    """

    def __init__(self, game=None, *, weighting="uniform", metadata=None, **kwargs):
        if weighting not in {"uniform", "linear"}:
            raise ValueError("weighting must be uniform or linear")
        config = dict(SELECTED_SD_CFR_KWARGS)
        config.update(kwargs)
        for key, required in (("policy_training_mode", "disabled"),
                              ("collect_strategy_replay", False),
                              ("compute_exploitability", False)):
            if config[key] != required:
                raise ValueError(f"Standalone SD-CFR requires {key}={required!r}")
        super().__init__(load_fhp_game() if game is None else game, **config)
        self.weighting = weighting
        self.archive = SDCFRArchive.from_solver(self, metadata={
            **dict(metadata or {}),
            "algorithm": "single_deep_cfr",
            "primary_weighting": weighting,
            "solver_config": config,
            "source_commit": "1669e5af4cbc88c648626148fd9c395c2e5d4583",
        })

    def solve(self, post_iteration_callback=None, post_player_update_callback=None,
              max_training_seconds=None):
        def capture(solver, player, iteration):
            self.archive.capture_from_solver(solver, player, iteration)
            if post_player_update_callback is not None:
                post_player_update_callback(solver, player, iteration)

        result = super().solve(
            post_iteration_callback=post_iteration_callback,
            post_player_update_callback=capture,
            max_training_seconds=max_training_seconds,
        )
        self.archive.validate()
        # Do not return the unused compatibility network as a deployable policy.
        result.policy_network = None
        return result

    def make_policy(self, *, seed=None, checkpoint=None):
        """A trajectory policy: call resample_episode() before EVERY new hand."""
        return SampledSDCFRPolicy(self._game, self.archive, seed=seed,
                                 weighting=self.weighting, checkpoint=checkpoint)

    def save_archive(self, path):
        """Save playable historical networks, without replay or optimiser state."""
        self.archive.metadata["completed_iterations"] = self._iteration - 1
        self.archive.metadata["nodes_touched"] = self._nodes_touched
        return self.archive.save(path)

    def action_probabilities(self, state, player_id=None):
        raise RuntimeError("SD-CFR has no average-policy network; use make_policy()")

    def save_policy_snapshot(self, *args, **kwargs):
        raise RuntimeError("SD-CFR must use save_archive(), not a Deep CFR policy snapshot")

    def extract_full_model(self, *args, **kwargs):
        raise RuntimeError(
            "A conventional Deep CFR checkpoint omits the SD-CFR archive. "
            "Use save_archive() for play; resumable training export is not implemented."
        )

    def load_full_model(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot resume standalone SD-CFR from a conventional Deep CFR checkpoint: "
            "its historical advantage archive is missing."
        )


__all__ = ["SELECTED_SD_CFR_KWARGS", "SingleDeepCFRSolver"]
