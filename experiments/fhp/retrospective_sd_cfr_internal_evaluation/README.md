# Retrospective SD-CFR internal head-to-head league

Evaluation only: no training, refitting, archive thinning, LBR, or source-output
modification. Every policy is played as its complete uniform historical SD-CFR
trajectory mixture.

## Questions and frozen sources

| Source | Run | Saved checkpoints used |
| --- | --- | --- |
| Exp2 raw inputs | `sdcfr2-24h-20261002-003338` | 24h |
| Exp3 structured inputs | `sdcfr3-24h-20261002-010643` | 24h |
| Exp4 sequential n2-standard-16 | `sdcfr4-vm16-20261002-095614` | 24h |
| Exp5 parallel traversal | `sdcfr5-par8-20261002-102757` | 24h |
| Exp6 48-hour parallel traversal | `sdcfr6-48h-20261002-161544` | 24/30/36/42/48h |
| Exp7 distributed fitting | `sdcfr7-distfit-20261003-172011` | 24h |

The eight prespecified comparisons are:

1. Exp3 24h minus Exp2 24h: structured representation.
2. Exp5 24h minus Exp4 24h: parallel traversal at equal active time.
3. Exp6 24h minus Exp5 24h: repeatability of the parallel configuration.
4. Exp7 24h minus Exp6 24h: distributed fitting at equal active time.
5. Exp7 24h minus Exp6 30h: the nearest saved node-budget bracket.
6. Exp6 36h minus Exp7 24h.
7. Exp6 42h minus Exp7 24h.
8. Exp6 48h minus Exp7 24h.

Positive reported values always favour the first policy named in a comparison.
No transitive ranking is inferred from comparisons that were not played.

## Evaluation and inference

Every comparison evaluates all nine training-seed combinations. Each cell uses
50,000 duplicate deal pairs with seats swapped: **3.6 million duplicate pairs
and 7.2 million hands** in total. Chance, action, and historical-network sampling
use split random streams. Production, smoke, and timing-pilot seeds are disjoint.

Reports retain per-cell finite-hand Monte Carlo uncertainty. Pointwise 95%
training-seed intervals use 10,000 independent bootstrap resamples of the three
row seeds and three column seeds; nine cells are not treated as nine independent
training runs. Same-seed values are descriptive. No multiplicity-adjusted claim
or automatic model promotion is made. Head-to-head strength is not exploitability
or proof of Nash convergence.

The job validates the exact source commits, experiment identities, game,
configuration, encoders, VM contracts, checkpoint schedules, checkpoint hashes,
and immutable archive chunks before play. Only playable archives and manifests
are downloaded; Exp6 replay/optimizer training states are excluded.

## Cloud launch

The evaluation uses one `n2-standard-8` VM with eight single-threaded evaluator
processes and no GPU. It first runs all-cell smoke checks, then a real-checkpoint
timing pilot. Production starts only when a conservative 2x estimate plus a
ten-minute allowance fits the 12-hour evaluation budget. Results are sharded,
uploaded every five minutes, integrity checked, and resumable under the same
run ID and code revision.

After committing and pushing this implementation:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SD_BUCKET="$BUCKET"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="FULL_PUSHED_COMMIT_SHA"
export RUN_ID="fhp-sdcfr-h2h-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=12

bash gcp/run_sd_cfr_internal_head_to_head.sh run
```

The six source run IDs default to the frozen runs above. They may be supplied
explicitly as `SD_EXP2_RUN_ID` through `SD_EXP7_RUN_ID`, but the evaluator still
requires the exact source contracts and commits. Use a new output run ID for a
different protocol or source cohort.

Useful commands:

```bash
bash gcp/run_sd_cfr_internal_head_to_head.sh status
bash gcp/run_sd_cfr_internal_head_to_head.sh dry-run --output /tmp/sdcfr-h2h-job.json

# Resume completed shards after an interrupted/failed evaluation VM.
bash gcp/run_sd_cfr_internal_head_to_head.sh resume
```

## Outputs and download

Outputs are stored under `$BUCKET/$RUN_ID/analysis/`:

- `summary.json`, `comparison_summary.csv`, and `matchups.csv`;
- `checkpoint_index.csv` and `evaluation_manifest.json`;
- `internal_comparison_forest.png`;
- `exp6_long_horizon_vs_exp7.png`;
- `exp6_48h_vs_exp7_24h_heatmap.png`;
- `timing_pilot.json`, resumable `task_results/`, and completion markers.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='.*(task_results|profile_tasks|smoke_tasks)/.*' \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

## Local verification

```bash
python -m pytest -q tests/test_sd_cfr_internal_head_to_head.py
```
