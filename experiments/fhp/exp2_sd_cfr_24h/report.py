"""Compact analysis tables and plots; training seeds are the inferential unit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from deep_cfr_poker.sd_cfr_disk import write_json
from .train import write_csv
from . import config as default_experiment


def summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Missing/non-finite results")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    margin = float(t.ppf(.975, len(values) - 1)) * se if se is not None else None
    return dict(n=len(values), mean=mean, se=se,
                ci95_low=mean - margin if margin is not None else None,
                ci95_high=mean + margin if margin is not None else None)


def evaluation_report(results, sd, output, *, smoke=False, include_lbr=True, has_comparison=False):
    output = Path(output)
    experiment_ids = {row["experiment"] for row in sd}
    if len(experiment_ids) != 1:
        raise ValueError("Expected one standalone experiment")
    experiment_id = experiment_ids.pop()
    if any(item["task"]["kind"] not in {"rule", "lbr", "temporal"} for item in results):
        raise ValueError("Only standalone SD-CFR evaluation tasks are supported")
    kinds = {item["task"]["kind"] for item in results}
    expected_kinds = {"rule", "temporal", "lbr"} if include_lbr else {"rule", "temporal"}
    if kinds != expected_kinds:
        raise ValueError("Evaluation results do not match the requested LBR scope")
    if not include_lbr and any((output / name).exists() for name in ("lbr_by_seed.csv", "lbr_aggregate.csv")):
        raise ValueError("Use a separate no-LBR output directory; preserve existing LBR results")
    rows = [dict(**{k: v for k, v in item["task"].items() if not k.startswith("path_")},
                 **item["result"], evaluation_seconds=item["elapsed_seconds"], experiment=experiment_id)
            for item in results]
    write_csv(output / "evaluation_tasks.csv", rows)
    rule = [r for r in rows if r["kind"] == "rule"]
    temporal = [r for r in rows if r["kind"] == "temporal"]
    grouped = defaultdict(list)
    for row in rows:
        if row["kind"] == "lbr":
            grouped[row["training_seed"], row["training_hours"]].append(row)
    lbr = []
    for (seed, hour), shards in sorted(grouped.items()):
        counts = np.array([s["num_deals"] for s in shards])
        lbr.append(dict(experiment=experiment_id, training_seed=seed, training_hours=hour,
                        kind="lbr", num_deals=int(counts.sum()),
                        mean_mbb_per_hand=float(np.average([s["mean_mbb_per_hand"] for s in shards], weights=counts)),
                        interpretation="sampled_LBR_value_not_exact_exploitability"))
    write_csv(output / "rule_agent_by_seed.csv", rule)
    if include_lbr:
        write_csv(output / "lbr_by_seed.csv", lbr)
    agent_groups = defaultdict(list)
    for row in rule:
        agent_groups[row["experiment"], row["training_hours"], row["opponent"]].append(row["mean_mbb_per_hand"])
    write_csv(output / "rule_agent_aggregate.csv", [dict(experiment=e, training_hours=h, opponent=o, **summary(v))
              for (e, h, o), v in sorted(agent_groups.items())])
    write_csv(output / "temporal_crossplay_by_seed.csv", temporal)
    rule_seed = defaultdict(list)
    for r in rule:
        rule_seed[r["experiment"], r["training_seed"], r["training_hours"]].append(r["mean_mbb_per_hand"])
    mean_rule = [dict(experiment=e, training_seed=s, training_hours=h, mean_mbb_per_hand=float(np.mean(v)))
                 for (e, s, h), v in sorted(rule_seed.items())]
    write_csv(output / "rule_agent_mean_by_seed.csv", mean_rule)
    indexes = {(r["experiment"], r["seed"], r["training_hours"]): r for r in sd}
    aggregates = []
    for metric, data in (("rule_agent_mean", mean_rule), ("lbr", lbr)):
        groups = defaultdict(list)
        for row in data:
            groups[row["experiment"], row["training_hours"]].append(row)
        for (experiment, hour), values in sorted(groups.items()):
            if len({r["training_seed"] for r in values}) != len(values):
                raise ValueError("Repeated training seed in aggregate")
            aggregates.append(dict(metric=metric, experiment=experiment, training_hours=hour,
                mean_nodes=float(np.mean([indexes[experiment, r["training_seed"], hour]["nodes_touched"] for r in values])),
                **summary([r["mean_mbb_per_hand"] for r in values])))
    write_csv(output / "quality_aggregate.csv", aggregates)
    if include_lbr:
        write_csv(output / "lbr_aggregate.csv", [row for row in aggregates if row["metric"] == "lbr"])
    write_csv(output / "rule_agent_mean_aggregate.csv", [row for row in aggregates if row["metric"] == "rule_agent_mean"])
    temporal_groups = defaultdict(list)
    for row in temporal:
        temporal_groups[row["earlier_hours"], row["training_hours"]].append(row["mean_mbb_per_hand"])
    write_csv(output / "temporal_crossplay_aggregate.csv", [dict(earlier_hours=e, later_hours=l, **summary(v))
              for (e, l), v in sorted(temporal_groups.items())])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4))
    pairs = sorted(temporal_groups)
    stats = [summary(temporal_groups[pair]) for pair in pairs]
    axis.errorbar(range(len(pairs)), [s["mean"] for s in stats],
                  yerr=[s["se"] or 0 for s in stats], fmt="o")
    axis.axhline(0, color="gray", linestyle="--")
    axis.set(xticks=range(len(pairs)), xticklabels=[f"{l}h vs {e}h" for e, l in pairs],
             ylabel="mbb/hand; mean ± one training-seed SE", title="Later versus earlier SD-CFR policies")
    if len(pairs) > 6:
        fig.set_size_inches(max(10, len(pairs) * 0.42), 5)
        axis.tick_params(axis="x", labelrotation=60, labelsize=8)
    if smoke:
        fig.suptitle("SMOKE TEST — short budgets, nominal checkpoint labels only")
    fig.tight_layout()
    fig.savefig(output / "temporal_head_to_head.png", dpi=180)
    plt.close(fig)
    panels = [("rule_agent_mean", "Five-agent mean (higher better)")]
    if include_lbr:
        panels.append(("lbr", "LBR value (lower better)"))
    for xkey, label in (("training_hours", "Active training hours"), ("mean_nodes", "Mean training nodes touched")):
        fig, axes = plt.subplots(1, len(panels), figsize=(11 if include_lbr else 7, 4), squeeze=False)
        for axis, (metric, title) in zip(axes.flat, panels):
            for experiment in sorted({r["experiment"] for r in aggregates}):
                selected = sorted([r for r in aggregates if r["metric"] == metric and r["experiment"] == experiment],
                                  key=lambda r: r[xkey])
                axis.errorbar([r[xkey] for r in selected], [r["mean"] for r in selected],
                              yerr=[r["se"] or 0 for r in selected], label=experiment, marker="o")
            axis.set(xlabel=label, ylabel="mbb/hand; mean ± one training-seed SE", title=title)
            axis.legend()
        if smoke:
            fig.suptitle("SMOKE TEST — short budgets, not production performance")
        fig.tight_layout()
        fig.savefig(output / f"policy_quality_by_{xkey}.png", dpi=180)
        plt.close(fig)
    lbr_note = ("LBR is a sampled lower-bound diagnostic, not exact exploitability or a convergence certificate.\n"
                if include_lbr else
                "LBR was deliberately omitted; no exploitability or exploiter estimate is reported.\n")
    scope = ("Routine evaluation tables; see exp7_vs_exp5_* and comparison_interpretation.txt for the fitting comparison.\n"
             if has_comparison else "Standalone SD-CFR evaluation; no cross-algorithm comparisons are included.\n")
    (output / "interpretation.txt").write_text(
        scope +
        "Error bars use independent training seeds, not hands; temporal matchups are paired within seed.\n"
        + lbr_note +
        "Rule-agent and temporal results measure playing strength against those opponents, not Nash convergence.\n"
        "Active time excludes checkpoint overhead. Node curves join observed checkpoint means.\n"
        "Playable checkpoints and evaluator provenance are retained for later comparative/exploiter analysis.\n")


def training_report(source, output, *, experiment=default_experiment):
    from .evaluate import checkpoint_index
    rows = checkpoint_index(source, experiment=experiment)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "checkpoint_index.csv", rows)
    aggregates = []
    for hour in sorted({r["training_hours"] for r in rows}):
        subset = [r for r in rows if r["training_hours"] == hour]
        aggregates.append(dict(training_hours=hour, **summary([r["nodes_touched"] for r in subset])))
    write_csv(output / "nodes_by_training_time.csv", aggregates)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4))
    for seed in sorted({r["seed"] for r in rows}):
        subset = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["training_hours"])
        axis.plot([r["active_seconds"] / 3600 for r in subset], [r["nodes_touched"] / 1e6 for r in subset],
                  marker="o", label=f"Seed {seed}")
    axis.set(xlabel="Actual active training hours", ylabel="Training nodes (millions)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_throughput.png", dpi=180)
    plt.close(fig)
    write_json(output / "training_summary.json", dict(status="complete", seeds=len({r["seed"] for r in rows}), checkpoints=len(rows),
                                                      exact_exploitability=False))


def main(*, experiment=default_experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_report(args.source, args.output, experiment=experiment)


if __name__ == "__main__":
    main()
