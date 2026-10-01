"""Compact analysis tables and plots; training seeds are the inferential unit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

from deep_cfr_poker.sd_cfr_disk import write_json
from .train import write_csv


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


def cluster_crossplay(rows):
    left = sorted({r["training_seed"] for r in rows})
    right = sorted({r["comparator_seed"] for r in rows})
    lookup = {(r["training_seed"], r["comparator_seed"]): r["mean_mbb_per_hand"] for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(left) * len(right):
        raise ValueError("Expected complete unique cross-seed matrix")
    values = np.array([[lookup[a, b] for b in right] for a in left])
    rng = np.random.default_rng(20260922)
    a = rng.integers(len(left), size=(10000, len(left)))
    b = rng.integers(len(right), size=(10000, len(right)))
    draws = values[a[:, :, None], b[:, None, :]].mean(axis=(1, 2))
    interval = np.quantile(draws, [.025, .975])
    return dict(mean_mbb_per_hand=float(values.mean()), ci95_low=float(interval[0]),
                ci95_high=float(interval[1]), sd_training_seeds=len(left), ucv_training_seeds=len(right),
                matchups=len(rows), positive_matchups=int((values > 0).sum()),
                bootstrap_draws=10000, inference="exploratory_two_way_training_seed_cluster_bootstrap")


def evaluation_report(results, sd, ucv, output, *, reference_root=None, smoke=False):
    output = Path(output)
    rows = [dict(**{k: v for k, v in item["task"].items() if not k.startswith("path_")},
                 **item["result"], evaluation_seconds=item["elapsed_seconds"], experiment="sd_cfr_exp2")
            for item in results]
    write_csv(output / "evaluation_tasks.csv", rows)
    rule = [r for r in rows if r["kind"] == "rule"]
    temporal = [r for r in rows if r["kind"] == "temporal"]
    direct = [r for r in rows if r["kind"] == "direct"]
    grouped = defaultdict(list)
    for row in rows:
        if row["kind"] == "lbr":
            grouped[row["training_seed"], row["training_hours"]].append(row)
    lbr = []
    for (seed, hour), shards in sorted(grouped.items()):
        counts = np.array([s["num_deals"] for s in shards])
        lbr.append(dict(experiment="sd_cfr_exp2", training_seed=seed, training_hours=hour,
                        kind="lbr", num_deals=int(counts.sum()),
                        mean_mbb_per_hand=float(np.average([s["mean_mbb_per_hand"] for s in shards], weights=counts)),
                        interpretation="sampled_LBR_value_not_exact_exploitability"))
    if reference_root:
        for name, destination in (("rule_agent_by_seed.csv", rule), ("lbr_by_seed.csv", lbr)):
            with (Path(reference_root) / name).open() as stream:
                for row in csv.DictReader(stream):
                    if row["experiment"] == "exp1":
                        destination.append(dict(row, experiment="ucv_exp1",
                                                training_seed=int(row["training_seed"]),
                                                training_hours=int(row["training_hours"]),
                                                mean_mbb_per_hand=float(row["mean_mbb_per_hand"])))
    write_csv(output / "rule_agent_by_seed.csv", rule)
    write_csv(output / "lbr_by_seed.csv", lbr)
    agent_groups = defaultdict(list)
    for row in rule:
        agent_groups[row["experiment"], row["training_hours"], row["opponent"]].append(row["mean_mbb_per_hand"])
    write_csv(output / "rule_agent_aggregate.csv", [dict(experiment=e, training_hours=h, opponent=o, **summary(v))
              for (e, h, o), v in sorted(agent_groups.items())])
    write_csv(output / "temporal_crossplay_by_seed.csv", temporal)
    if direct:
        write_csv(output / "direct_crossplay_by_seed.csv", direct)
        write_json(output / "direct_crossplay_summary.json", cluster_crossplay(direct))
    rule_seed = defaultdict(list)
    for r in rule:
        rule_seed[r["experiment"], r["training_seed"], r["training_hours"]].append(r["mean_mbb_per_hand"])
    mean_rule = [dict(experiment=e, training_seed=s, training_hours=h, mean_mbb_per_hand=float(np.mean(v)))
                 for (e, s, h), v in sorted(rule_seed.items())]
    write_csv(output / "rule_agent_mean_by_seed.csv", mean_rule)
    indexes = {(r["experiment"], r["seed"], r["training_hours"]): r for r in sd + ucv}
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
    if direct:
        matrix = np.array([[next(r["mean_mbb_per_hand"] for r in direct
                                  if r["training_seed"] == a and r["comparator_seed"] == b)
                            for b in sorted({r["comparator_seed"] for r in direct})]
                           for a in sorted({r["training_seed"] for r in direct})])
        fig, axis = plt.subplots(figsize=(6, 4))
        limit = max(float(np.abs(matrix).max()), 1.0)
        plot = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.1f}", ha="center", va="center")
        axis.set(xlabel="UCV training seed", ylabel="SD-CFR training seed",
                 xticks=range(matrix.shape[1]), yticks=range(matrix.shape[0]),
                 title="24-hour head-to-head; positive favours SD-CFR")
        fig.colorbar(plot, ax=axis, label="mbb/hand")
        if smoke:
            fig.suptitle("SMOKE TEST — not production performance")
        fig.tight_layout()
        fig.savefig(output / "sd_cfr_vs_ucv_head_to_head.png", dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    pairs = sorted(temporal_groups)
    stats = [summary(temporal_groups[pair]) for pair in pairs]
    axis.errorbar(range(len(pairs)), [s["mean"] for s in stats],
                  yerr=[s["se"] or 0 for s in stats], fmt="o")
    axis.axhline(0, color="gray", linestyle="--")
    axis.set(xticks=range(len(pairs)), xticklabels=[f"{l}h vs {e}h" for e, l in pairs],
             ylabel="mbb/hand; mean ± one training-seed SE", title="Later versus earlier SD-CFR policies")
    if smoke:
        fig.suptitle("SMOKE TEST — short budgets, nominal checkpoint labels only")
    fig.tight_layout()
    fig.savefig(output / "temporal_head_to_head.png", dpi=180)
    plt.close(fig)
    for xkey, label in (("training_hours", "Active training hours"), ("mean_nodes", "Mean training nodes touched")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, metric, title in zip(axes, ("rule_agent_mean", "lbr"),
                                       ("Five-agent mean (higher better)", "LBR value (lower better)")):
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
    (output / "interpretation.txt").write_text(
        "Independent historical cohorts, not paired training. Error bars use training seeds, not hands.\n"
        "LBR is a sampled lower-bound diagnostic, not exact exploitability or a convergence certificate.\n"
        "Active time excludes checkpoint overhead; UCV also excluded its average-policy fitting.\n"
        "Node curves join checkpoint means; algorithms count different interaction work.\n"
        "Direct crossplay intervals resample each method's seeds independently, not the nine shared-policy cells.\n")


def training_report(source, output):
    from .evaluate import checkpoint_index
    rows = checkpoint_index(source)
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
    write_json(output / "training_summary.json", dict(status="complete", seeds=3, checkpoints=12,
                                                      exact_exploitability=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_report(args.source, args.output)
