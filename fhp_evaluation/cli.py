"""Command-line entry point for duplicate, benchmark, and LBR evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .benchmark import evaluate_against_published_agents
from .duplicate import evaluate_duplicate_match
from .game import load_fhp_game
from .lbr import LBRConfig, evaluate_lbr
from .loaders import load_checkpoint_policy


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write(result, output: str | None) -> None:
    text = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    duplicate = subparsers.add_parser("duplicate", help="both-seat duplicate match")
    duplicate.add_argument("policy_a")
    duplicate.add_argument("policy_b")
    duplicate.add_argument("--deals", type=int, default=10_000)
    duplicate.add_argument("--seed", type=int, default=0)
    duplicate.add_argument("--output")

    benchmark = subparsers.add_parser("benchmark", help="checkpoint versus five agents")
    benchmark.add_argument("policy")
    benchmark.add_argument("--deals", type=int, default=10_000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output")

    lbr = subparsers.add_parser("lbr", help="both-seat Local Best Response lower bound")
    lbr.add_argument("policy")
    lbr.add_argument("--deals", type=int, default=1_000)
    lbr.add_argument("--preflop-rollouts", type=int, default=4096)
    lbr.add_argument("--seed", type=int, default=0)
    lbr.add_argument("--output")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    game = load_fhp_game()
    if args.command == "duplicate":
        left = load_checkpoint_policy(game, args.policy_a)
        right = load_checkpoint_policy(game, args.policy_b)
        result = evaluate_duplicate_match(
            game,
            left,
            right,
            num_deals=args.deals,
            seed=args.seed,
            policy_a_name=Path(args.policy_a).stem,
            policy_b_name=Path(args.policy_b).stem,
        ).to_dict()
        result["policy_a_sha256"] = left.sha256
        result["policy_b_sha256"] = right.sha256
    elif args.command == "benchmark":
        target = load_checkpoint_policy(game, args.policy)
        result = evaluate_against_published_agents(
            game,
            target,
            num_deals=args.deals,
            seed=args.seed,
            target_name=Path(args.policy).stem,
        )
        result["target"] = str(Path(args.policy).resolve())
        result["target_sha256"] = target.sha256
    else:
        target = load_checkpoint_policy(game, args.policy)
        result = evaluate_lbr(
            game,
            target,
            num_deals=args.deals,
            seed=args.seed,
            config=LBRConfig(
                preflop_rollout_samples=args.preflop_rollouts,
                seed=args.seed,
            ),
            target_name=Path(args.policy).stem,
        )
        result["target"] = str(Path(args.policy).resolve())
        result["target_sha256"] = target.sha256
    _write(result, args.output)


if __name__ == "__main__":
    main()
