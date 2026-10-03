"""Read-only old/new SD-CFR LBR correctness and timing audit.

Use --full-lbr only for small diagnostic archives: the scalar baseline is the
very slow implementation being investigated. Production profiles instead gate
the complete archive on selected queries, then time the new full LBR alone.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .game import load_fhp_game
from .sd_cfr_disk import DiskArchiveReader, DiskBehaviouralPolicy, write_json, sha256
from .sd_cfr_lbr import BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy
from fhp_evaluation.lbr import LocalBestResponsePolicy, LBRConfig, _policy_probability


HISTORIES = (
    (0, 4, 8, 12), (0, 4, 8, 12, 2), (0, 4, 8, 12, 2, 2, 2),
    (0, 4, 8, 12, 1, 1, 16, 20, 32),
    (0, 4, 8, 12, 1, 1, 16, 20, 32, 2),
)


def states_for_audit(game):
    states = []
    for history in HISTORIES:
        state = game.new_initial_state()
        for action in history:
            state.apply_action(action)
        states.append(state)
    return states


def validate_queries(reader, game, *, batched=None, tolerance=2e-6):
    """Fail closed on changed probabilities/own reach, using EVERY saved model."""
    scalar = DiskBehaviouralPolicy(reader, game)
    batched = batched or BatchedDiskBehaviouralPolicy(reader, game)
    states = states_for_audit(game)
    maximum_probability_error, maximum_reach_error = 0., 0.
    for player in (0, 1):
        selected = [s for s in states if s.current_player() == player]
        reaches, probabilities = batched.batch_reach_and_probabilities(selected, player)
        for state, reach, probability in zip(selected, reaches, probabilities):
            reference = scalar.action_probabilities(state)
            old = np.asarray([reference[a] for a in state.legal_actions()])
            new = probability[state.legal_actions()]
            maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(old-new))))
            cursor, likelihood = game.new_initial_state(), 1.
            for action in state.history():
                if not cursor.is_chance_node() and cursor.current_player() == player:
                    likelihood *= _policy_probability(scalar, cursor, player, action)
                cursor.apply_action(action)
            maximum_reach_error = max(maximum_reach_error, abs(likelihood-reach))
    report = dict(models_per_player=reader.count, device=str(batched.device), queries=len(states), tolerance=tolerance,
                  max_probability_error=maximum_probability_error, max_reach_error=maximum_reach_error,
                  passed=max(maximum_probability_error, maximum_reach_error) <= tolerance)
    if not report["passed"]:
        raise RuntimeError(f"Exact batched SD-CFR LBR failed its numerical gate: {report}")
    return report


def audit(path, *, full_lbr=False, rollouts=4096):
    game = load_fhp_game()
    reader = DiskArchiveReader(path, game)
    report = dict(archive_sha256=sha256(path), archive=str(path), models_per_player=reader.count,
                  feature_encoder=reader.contract.get("metadata", {}).get("feature_encoder"),
                  torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                  correctness=validate_queries(reader, game), full_lbr_comparisons=[])
    if full_lbr:
        # Warm up torch's functional/vmap machinery before the timed regions.
        for state in states_for_audit(game):
            rows, hypotheses = [], []
            for policy_type, lbr_type in ((DiskBehaviouralPolicy, LocalBestResponsePolicy),
                                         (BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy)):
                policy = policy_type(reader, game)
                lbr = lbr_type(game, policy, config=LBRConfig(seed=21760922, preflop_rollout_samples=rollouts))
                start = time.perf_counter()
                ranges = lbr._opponent_range(state, state.current_player())
                folded = lbr._raise_fold_range(ranges, state.current_player())[0] if 2 in state.legal_actions() else None
                query_seconds = time.perf_counter() - start
                # Range/probability caches are warm here for BOTH implementations.
                action = lbr.action_probabilities(state)
                rows.append(dict(backend=policy_type.__name__, query_seconds=query_seconds,
                                 fold_probability=folded, action=action))
                hypotheses.append({h.hand: h.weight for h in ranges})
            error = max(abs(hypotheses[0].get(h, 0)-hypotheses[1].get(h, 0))
                        for h in hypotheses[0].keys() | hypotheses[1].keys())
            assert error <= 2e-6, error
            assert rows[0]["action"] == rows[1]["action"], rows
            if rows[0]["fold_probability"] is not None:
                assert abs(rows[0]["fold_probability"] - rows[1]["fold_probability"]) <= 2e-6
            record = dict(history=state.history(), max_range_error=error,
                          speedup=rows[0]["query_seconds"] / rows[1]["query_seconds"], backends=rows)
            report["full_lbr_comparisons"].append(record)
            print(record, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full-lbr", action="store_true")
    parser.add_argument("--rollouts", type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(1)
    write_json(args.output, audit(args.archive, full_lbr=args.full_lbr, rollouts=args.rollouts))


if __name__ == "__main__":
    main()
