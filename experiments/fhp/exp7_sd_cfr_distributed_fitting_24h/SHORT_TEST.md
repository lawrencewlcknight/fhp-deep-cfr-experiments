# Experiment 7: short distributed-fitting screen

This standalone engineering test compares Experiment 5's central fitting with
Experiment 7's synchronous eight-worker fitting on **one n2-standard-16 VM**.
Both arms retain eight traversal workers, one CPU thread per worker, and the
same learner hyperparameters. Arms run sequentially, never concurrently.
It does not start a controller, long training, or a policy-evaluation suite.

## Budget and design

The nominal workload is approximately **30 active minutes**:

1. Up to approximately five minutes for worker startup, real replay collection,
   warmup, and tight single-update gradient/Adam checks for both players.
2. Twenty minutes of paired frozen-replay fits using production-sized global
   batches of 2,048 and 200 updates. Every pair starts with identical weights,
   nonempty Adam state and sampling seed. Both execution orders are tested for
   each player. The same eight actors remain available but idle during central
   fits; all replay preparation, transfers and synchronization are timed.
3. Approximately five active minutes of actual traversal/fitting/archive-capture
   loops. A one-iteration pair estimates runtime, then a reversed-order pair
   runs an equal number of outer iterations and root traversals in each arm.
   Actual nodes may differ as floating-point drift changes action sampling;
   both matched-work time and observed nodes/second are reported.

Stages stop at complete pair boundaries, so may overrun their target. Warmup
may finish early. End-to-end worker startups and final archive validation are
reported outside active-loop timing. Allow roughly **45–60 elapsed minutes**
including provisioning, package installation and process startup; this is an
estimate, not a guaranteed deadline. The Batch safety limit is two hours with
**no automatic retries**. Only one source seed (default 0) is used: repeated
timings are not independent training seeds.

## Correctness and interpretation

Non-finite values, incorrect single-step gradients/Adam updates, different
sampling RNG streams, mutated frozen replay, failed worker synchronization or
invalid archive capture fail the test. The standalone benchmark still defaults
to a strict full-fit near-output gate. Following review of the completed cloud
screen, the separately approved [24-hour comparison](README.md) explicitly allows
accumulated drift while retaining all single-update correctness gates. This short
screen itself does not launch or automatically approve that follow-up.

Full 200-update parameter/logit/action-probability differences are recorded,
but do not abort this diagnostic. They can amplify despite correct individual
updates. A completed Batch job therefore means **the screen completed**, not
that long fits are equivalent or that the distributed learner is approved.
Check `numerical_drift_requires_review` as well as timing before promotion.

The frozen fixture targets 32,768 genuinely collected rows per player, or the
available rows within the warmup allocation (minimum one full global batch).
Rows are not tiled or duplicated to simulate a full reservoir. Actual occupancy
and the five-million-row production capacity are reported separately.
The end-to-end segment starts from empty replay in both arms. Thus this is an
early-run engineering screen, **not a full-reservoir memory benchmark or an
estimate of 24-hour throughput**, and it says nothing conclusive about poker
strength. No speed improvement is presumed. Review the evidence before deciding
whether a longer matched-seed quality/efficiency comparison is warranted.

## Launch after committing and pushing

From this repository with the usual `PROJECT_ID`, `REGION`, `BUCKET` and
`SA_EMAIL` already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr7-short-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp7_sd_cfr_short_test.sh run
```

Use `dry-run` instead of `run` to inspect the single-job JSON without submission.
Use `status` with the same RUN_ID to check the job. Submission checks that the
pinned commit contains the implementation and that the service account exists;
it does not create accounts or modify IAM. It needs no child-job permissions.

For a tiny local integration check with existing dependencies (not a timing
measurement), use `bash gcp/run_exp7_sd_cfr_short_test.sh smoke-local`.

## Retained outputs

Only small analysis and metadata files are uploaded to
`gs://BUCKET/RUN_ID/analysis/`, periodically and at exit:

- `manifest.json`, `warmup.json`, `update_checks.json`: provenance and hard checks.
- `fitting_timings.csv`, `fitting_differences.csv`, `fitting_summary.json`:
  paired timings, compute/communication breakdown and accumulated numerical drift.
- `end_to_end_timings.csv`, `end_to_end_trajectory.csv`: actual training-loop timing
  and node counts, startup and archive validation.
- `summary.json`, `progress.json`, `SUCCESS.json` or `FAILURE.json`.

The summary's speedup is central seconds divided by distributed seconds for
matched work: greater than 1 means faster distributed execution. Worker times
overlap and must not be added to obtain elapsed time. Startup is separate.
Temporary playable archives are validated locally and discarded; no reservoir,
full training state, or model artifact is retained by this diagnostic.
