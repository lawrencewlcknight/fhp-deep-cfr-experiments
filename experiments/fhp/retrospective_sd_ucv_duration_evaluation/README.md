# SD-CFR versus UCV-ESCHER at 24 and 48 active hours

Evaluation only: no training, policy fitting, archive thinning, LBR or repeated
rule-agent evaluation. The four trained runs and their source outputs remain
read-only.

## Frozen sources and questions

| Role | Run | Bucket |
|---|---|---|
| SD-CFR 24h | Exp5 `sdcfr5-par8-20261002-102757` | `gs://clever-overview-399515-fhp-deep-cfr-results` |
| UCV-ESCHER 24h | Exp10 `exp10-features-20261001-161740` | `gs://clever-overview-399515-fhp-escher-results` |
| SD-CFR 48h | Exp6 `sdcfr6-48h-20261002-161544` | `gs://clever-overview-399515-fhp-deep-cfr-results` |
| UCV-ESCHER 48h | Exp16 `exp16-feat48-20261004-182051` | `gs://clever-overview-399515-fhp-escher-results` |

The two prespecified primary questions are whether SD-CFR Exp5 beats UCV Exp10
at 24 active hours and whether SD-CFR Exp6 beats UCV Exp16 at 48 active hours.
Experiment 16 is validated as the exact continuation of the three Experiment 10
states: its lineage file must identify the frozen 24h source and its 6--24h
checkpoint hashes, node counts and iterations must equal Experiment 10.

Each comparison uses all **nine cross-seed pairings**. Matching integer seed
labels across algorithms is not treated as statistical pairing.

| Cohort | Comparison | SD hours | UCV hours | Duplicate pairs per seed pairing |
|---|---|---:|---:|---:|
| 24h | Primary endpoint | 24 | 24 | 100,000 |
| 24h | Equal-time trajectory | 6 / 12 / 18 | 6 / 12 / 18 | 50,000 each |
| 24h | Approximate node match | 6 | 12 | 50,000 |
| 24h | Approximate node match | 12 | 24 | 50,000 |
| 48h | Same-lineage bridge | 24 | 24 | 50,000 |
| 48h | Equal-time continuation | 30 / 36 / 42 | 30 / 36 / 42 | 50,000 each |
| 48h | Primary endpoint | 48 | 48 | 100,000 |
| 48h | Approximate node match | 24 | 48 | 50,000 |

Total: **108 matchup cells, 6.3 million duplicate pairs and 12.6 million
hands**. The approximate node comparisons were fixed before observing these
matchups. Expected mean-node brackets from the frozen indexes are roughly
87.4m versus 81.6m, 176.2m versus 160.6m, and 351.0m versus 318.3m. Equal active
time is not equal nodes, compute, deployment cost or billable cost.

## Policy fidelity and inference

- SD-CFR uses `DiskSampledPolicy`: independently select one historical network
  per player at the start of a hand and retain it for the hand. The complete
  uniform archive is used, not the final network or a behavioural average.
- UCV uses each saved deployed average-policy network through the separately
  pinned `LoadedFHPPolicy`; no critic or privileged training input is deployed.
- The common FHP game and duplicate evaluator use `seed_layout="split"`; every
  deal is replayed with seats swapped. Chance, action and SD-mixture randomness
  use separate streams.
- Values are mbb/hand, **positive favouring SD-CFR**. Duplicate pairs are the
  Monte Carlo units.
- Pointwise 95% intervals use 10,000 independent row/column training-seed
  bootstrap draws. The two primary endpoints additionally receive 97.5%
  Bonferroni-compatible intervals. Conditional finite-hand Monte Carlo errors
  are reported separately.
- The 48h report includes the change in the SD-minus-UCV cross-play margin from
  the Exp6/Exp16 24h bridge to 48h. This is not a causal or universal estimate
  of which algorithm learns faster.
- Three seeds per method support an exploratory comparison, not proof of a
  universal ranking, exact exploitability or convergence.

## Cloud execution

One small controller launches two independent, resumable `n2-standard-8`
evaluation jobs concurrently. Each uses eight single-threaded evaluation
processes and a real-checkpoint timing gate with a 2x margin. The 24h stage has
a 200 GiB disk; the larger 48h stage has 300 GiB. After both stages succeed, a
small aggregation job produces the combined report. No source policy files are
uploaded as outputs.

All stages use bounded package-install retries during VM startup: up to 30
attempts per apt command, with a 10-second dpkg-lock wait and a 10-second pause
between failures. This handles contention with unattended upgrades without
deleting locks or stopping the updater. Persistent failures retain their exit
status and are logged; the controller identifies the failed child job and state.

After committing and pushing the implementation:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SD_BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export UCV_BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="FULL_PUSHED_SD_CFR_COMMIT_SHA"
export RUN_ID="fhp-sd-ucv-duration-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=12

bash gcp/run_sd_ucv_duration_head_to_head.sh run
```

The source-run defaults are the four frozen runs above. They can be overridden
with `SD_EXP5_RUN_ID`, `SD_EXP6_RUN_ID`, `UCV_EXP10_RUN_ID` and
`UCV_EXP16_RUN_ID`. `UCV_REPO_REF` defaults to a pushed UCV commit containing
the Experiment 10/16 feature loader. The launcher validates all source markers
before submitting the controller.

Monitor or resume with the same run identity and refs:

```bash
bash gcp/run_sd_ucv_duration_head_to_head.sh status
bash gcp/run_sd_ucv_duration_head_to_head.sh resume
```

Resumption checksum-validates and reuses completed 5,000-pair shards. A
completed cohort is not rerun. Changing code, sources or protocol requires a
new `RUN_ID`.

Exception for the October 2026 startup-lock repair: the change is confined to
the Batch launcher, not the evaluation implementation or scientific identity.
After committing and pushing this repair, set `REPO_REF` to that full SD-CFR
commit and retain `RUN_ID=fhp-sd-ucv-duration-20261006-193151`, the original
source runs and `UCV_REPO_REF`. Run the `resume` command above. Both cohort
completion markers already exist, so only a new controller and aggregation job
are submitted; the completed 6.3 million duplicate pairs are not replayed.

## Outputs

Stage data is stored under `$BUCKET/$RUN_ID/stages/{24h,48h}/`. The combined
analysis is under `$BUCKET/$RUN_ID/analysis/`:

- `analysis_summary.md`, `summary.json`, `comparison_summary.csv`;
- both primary 3x3 heatmaps;
- equal-time trajectory and approximate-node figures for each cohort;
- `SUCCESS.json` with the two source-stage manifest hashes.

Download only the combined result:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

## Local tests

```bash
UCV_TEST_REPO=/absolute/path/to/fhp-ucv-escher-experiments \
  python -m pytest -q tests/test_sd_ucv_duration_head_to_head.py
```

Without `UCV_TEST_REPO`, the synthetic real-loader integration test is skipped;
the protocol, inference, integrity, reporting and cloud-plan tests still run.
