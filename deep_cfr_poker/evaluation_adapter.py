"""Deep CFR entry points into the shared FHP evaluation suite."""

from __future__ import annotations

from pathlib import Path
import sys

from .game import load_fhp_game


def _import_suite():
    try:
        import fhp_evaluation
    except ModuleNotFoundError:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "fhp-evaluation-suite"
            if (candidate / "fhp_evaluation").is_dir():
                sys.path.insert(0, str(candidate))
                import fhp_evaluation
                break
        else:
            raise ModuleNotFoundError(
                "Install the shared suite with: python -m pip install -e ../../fhp-evaluation-suite"
            ) from None
    return fhp_evaluation


def load_policy_for_evaluation(snapshot_path):
    _import_suite()
    from fhp_evaluation.loaders import load_checkpoint_policy

    game = load_fhp_game()
    return game, load_checkpoint_policy(game, snapshot_path)


def evaluate_checkpoint(
    snapshot_path,
    *,
    num_deals: int,
    seed: int,
    lbr_deals: int = 0,
    lbr_rollouts: int = 4096,
):
    _import_suite()
    from fhp_evaluation.benchmark import evaluate_against_published_agents
    from fhp_evaluation.lbr import LBRConfig, evaluate_lbr

    game, target = load_policy_for_evaluation(snapshot_path)
    result = evaluate_against_published_agents(
        game, target, num_deals=num_deals, seed=seed, target_name="deep_cfr"
    )
    if lbr_deals:
        result["lbr"] = evaluate_lbr(
            game,
            target,
            num_deals=lbr_deals,
            seed=seed,
            config=LBRConfig(preflop_rollout_samples=lbr_rollouts, seed=seed),
            target_name="deep_cfr",
        )
    return result
