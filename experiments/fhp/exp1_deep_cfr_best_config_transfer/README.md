# Experiment 1 — Best-configuration transfer to FHP

This experiment trains Deep CFR on the repository's canonical FHP game using
the approved best previously validated configuration. It is a single fixed
configuration evaluated across five reproducible training seeds, not an
ablation or hyperparameter search.

## Approved defaults

| Setting | Value |
| --- | --- |
| Seeds | `1234, 2025, 31415, 27182, 16180` |
| Iterations | `1050` |
| Traversals | `320` per traversing player per iteration |
| Policy network | `2x32` MLP |
| Advantage network | `8x32` residual LayerNorm centred-advantage MLP |
| Learning rate | `0.004`, constant |
| Advantage / strategy batch | `2048 / 1024` |
| Replay capacity | `5,000,000` |
| Advantage / policy steps | `200 / 200` |
| Policy fitting cadence | Every `10` iterations |
| Target processing | Standardise, epsilon `1e-6` |
| Replay / averaging weights | Uniform / uniform |
| Execution backend | Sequential |
| Policy snapshots | `100, 250, 500, 750, 1050` |
| Exact exploitability | Disabled |

Every snapshot coincides with a freshly fitted average-policy network. The
final checkpoint contains replay buffers and RNG state and can therefore be
large; this is intentional so that the approved run is resumable.

## Local smoke test

```bash
python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 2 \
  --evaluation-interval 1 \
  --checkpoint-schedule 1,2 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-train-every 1 \
  --output-root outputs/smoke
```

## Full run

```bash
python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run
```

## Google Cloud Batch

After publishing this local repository to the launcher's default repository URL
(or setting `REPO_URL` to its published location), set `PROJECT_ID`, `REGION`,
`BUCKET`, and `SA_EMAIL`:

```bash
JOB_NAME="fhp-deep-cfr-exp1-$(date +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_deep_cfr_best_config_transfer.run \
    --output-root outputs/cloud/$JOB_NAME" \
  "n2-highmem-8" \
  "345600" \
  "8000" \
  "64000" \
  "200" \
  "pd-balanced"
```

The 96-hour cap, 64 GiB memory request, and 200 GiB boot disk are conservative
starting resources for five sequential seeds and buffer-inclusive checkpoints.

## Outputs

- `run_manifest.json`: live status, exact game definition, configuration, and
  completed/failed seeds;
- `policy_snapshot_manifest.csv` and per-seed JSON manifests;
- lightweight policy snapshots under `snapshots/`;
- resumable final checkpoints under `checkpoints/`;
- `seed_summary.csv`, `checkpoint_curves.csv`, `aggregate_summary.json`, and
  `multiseed_curves.npz`.

No exact equilibrium-quality claim is made. The saved policies are intended
for a later sampled, seat-averaged head-to-head evaluation experiment.
