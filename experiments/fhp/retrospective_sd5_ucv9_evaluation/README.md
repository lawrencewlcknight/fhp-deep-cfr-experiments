# SD-CFR Experiment 5 versus UCV-ESCHER Experiment 9

Evaluation only: no training, policy refitting, LBR, or repeated rule-agent
evaluation. Neither trained model nor its source outputs are modified.

## Frozen sources and primary question

Does the deployed SD-CFR Experiment 5 policy beat the deployed UCV-ESCHER
Experiment 9 policy at the saved 24-active-hour boundary?

| Source | Run | Bucket |
| --- | --- | --- |
| SD-CFR Exp5 | `sdcfr5-par8-20261002-102757` | `gs://clever-overview-399515-fhp-deep-cfr-results` |
| UCV-ESCHER Exp9 | `exp9-cache24-20261001-132550` | `gs://clever-overview-399515-fhp-escher-results` |

Both contribute seeds 0, 1 and 2 and checkpoints at 6, 12, 18 and 24 hours.
All **nine cross-seed pairings** are evaluated at every comparison; matching
integer seed labels across different algorithms does not establish pairing.

| Comparison | SD hours | UCV hours | Duplicate pairs per seed pairing |
| --- | ---: | ---: | ---: |
| Primary endpoint | 24 | 24 | 100,000 |
| Same-time trajectory | 6 / 12 / 18 | 6 / 12 / 18 | 50,000 each |
| Approximate node match | 6 | 12 | 50,000 |
| Approximate node match | 12 | 24 | 50,000 |

Total: **54 matchup cells, 3.15 million duplicate pairs, 6.3 million hands**.
Node matches were selected before observing matchup results: approximately
87.4m versus 85.4m and 176.2m versus 169.9m mean nodes. The actual per-policy
node counts and relative differences are reported. No interpolated policy is
played, and these are not exactly matched-node or equal-compute comparisons.

## Policy fidelity and inference

- Use the bundled common FHP game and both-seat duplicate evaluator, with
  `seed_layout="split"`. Within a pair, chance deals are repeated with seats
  swapped. Chance, action and SD mixture sampling use separate RNG streams.
- SD-CFR uses `DiskSampledPolicy`: independently sample a historical network
  per player at the beginning of a hand, retain it throughout that hand, and
  use the complete uniform archive. This is the algorithm's playable trajectory
  mixture, not the final network or an unweighted behavioural average.
- UCV uses its saved deployed average-policy network through the separately
  pinned repository's `LoadedFHPPolicy`; no privileged critic inputs are used.
- Production source commits, configurations, encoder, game, completed-worker
  markers, checkpoint schedules, metadata, hashes and all archive chunks are
  validated. Chunk hashes are computed once per unique immutable chunk.
- Report **mbb/hand, positive favouring SD-CFR**. Duplicate pairs, not their
  individual hands, are the Monte Carlo sampling units.
- Report within-cell Monte Carlo errors separately from pointwise 95% training-
  seed cluster-bootstrap intervals. The latter use 10,000 draws, resampling
  the three row seeds and three column seeds independently; the nine cells
  are not nine independent trained models. Intervals condition on estimated
  cell means. The independent conditional aggregate Monte Carlo SE is also
  provided so finite-hand precision is visible.
- Three seeds per method support exploratory comparison, not a definitive
  universal ranking. There is no claim of exact exploitability or convergence.
  Active clocks have different exclusions (notably UCV policy fitting); archive
  and deployment costs differ too. This is not equal end-to-end training cost.

## Cloud job

One standard `n2-standard-8`, eight evaluation processes with one Torch/BLAS
thread each, no GPU. A 200 GiB boot disk holds the read-only policy inputs;
source model files are **not** re-uploaded as evaluation outputs. No automatic
task retries. Source and output buckets are independent parameters.

The VM executes a short smoke across every comparison, then a timing pilot
of 128 duplicate pairs for each of the 54 cells using the real full archives.
Production begins only if measured remaining work, with a **2x time margin**
and ten-minute allowance, fits the 12-hour evaluation budget. The existing
4–8-hour expectation is provisional; the pilot is the runtime evidence.
Setup and source download have an additional two hours in the Batch limit.
There is no fallback that silently reduces match counts or simplifies policies.

Completed 5,000-pair shards are saved atomically and uploaded every five
minutes, plus on normal/error exit. Resumption verifies source, code, runtime,
protocol and result checksums, skips intact completed shards, and profiles
the remaining work on the new VM. At the evaluation deadline no new shards
are started; already-running shards finish. A forced VM loss can require
repeating the last not-yet-uploaded shards, not the entire evaluation.

The service account needs read access to **both source buckets**, output-object
write access, and standard Batch logging/agent permissions. The launcher checks
source metadata access as the submitting user; the VM checks again under its
own service account. This single-job evaluator creates no child jobs and makes
no IAM changes.

After this implementation is committed and pushed:

```bash
export BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export SD_BUCKET="gs://clever-overview-399515-fhp-deep-cfr-results"
export UCV_BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="FULL_PUSHED_SD_CFR_COMMIT_SHA"
export RUN_ID="fhp-sd5-ucv9-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=12

bash gcp/run_sd5_ucv9_head_to_head.sh run
```

`PROJECT_ID` and `REGION` must already be set. Default source run IDs are the
ones above; `SD_EXP5_RUN_ID` and `UCV_EXP9_RUN_ID` can explicitly specify them.
`UCV_REPO_REF` defaults to the pinned loader commit
`1b61ebcabf3f3865ba32430b388fdbdf6d34297a`. The two overlapping Python
`experiments` packages are not installed together: SD is first on the import
path; only UCV's unique policy-loader modules are appended.

Use `status` to inspect, `dry-run --output /tmp/sd5-ucv9-job.json` to generate
Batch JSON without cloud writes, or `resume` after a failed/interrupted job
using the same `RUN_ID`, refs and sources. The launcher refuses simultaneous
jobs using the same output prefix. To change implementation or protocol,
choose a new output prefix; do not mix incompatible cached matches.

## Outputs and download

Everything is under `$BUCKET/$RUN_ID/analysis/`:

- `summary.json`, `comparison_summary.csv`: primary and secondary estimates,
  seed-cluster intervals and conditional Monte Carlo errors.
- `matchups.csv`: all 54 cells, per-seat values, actual budgets and sampling errors.
- `checkpoint_index.csv`, `evaluation_manifest.json`: source hashes and provenance.
- `final_head_to_head_heatmap.png`, `head_to_head_by_training_time.png`,
  `approximate_nodes.png`: the three comparison figures.
- `timing_pilot.json`, smoke/profile results, `task_results/`: auditable timing
  and resumable result shards. Only a complete production budget produces
  `SUCCESS.json`; smoke/pilot results never enter scientific summaries.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='.*(task_results|profile_tasks|smoke_tasks)/.*' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local tests

```bash
UCV_TEST_REPO=/absolute/path/to/fhp-ucv-escher-experiments \
  python -m pytest -q tests/test_sd5_ucv9_head_to_head.py
```

The optional integration tests build tiny synthetic policy checkpoints and
exercise both real loaders and multiprocessing, without training a model.
Without `UCV_TEST_REPO`, cross-repository integration tests are explicitly
skipped; the protocol, statistics, integrity and cloud-plan tests still run.
