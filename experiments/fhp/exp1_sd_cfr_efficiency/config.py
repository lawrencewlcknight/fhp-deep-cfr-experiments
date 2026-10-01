"""Fixed-work benchmark: no hyperparameter tuning or policy-quality claim."""

from deep_cfr_poker.single_solver import SELECTED_SD_CFR_KWARGS

EXPERIMENT_NAME = "exp1_sd_cfr_efficiency"
ARMS = ("reference", "scripted", "scripted_packed")
DEFAULT_SEEDS = (1234, 2025, 31415)
DEFAULT_REPEATS = 3
ATOL = 1e-6
RTOL = 1e-5


def solver_config(smoke=False):
    config = dict(SELECTED_SD_CFR_KWARGS)
    # Small reservoir for a short benchmark, identical in every arm. Network,
    # traversals, gradient steps, batches, LR and diagnostics retain the recipe.
    config.update(num_iterations=6, memory_capacity=100_000)
    if smoke:
        config.update(num_iterations=3, num_traversals=4, memory_capacity=16,
                      advantage_network_train_steps=2, batch_size_advantage=8,
                      evaluation_interval=1)
    return config
