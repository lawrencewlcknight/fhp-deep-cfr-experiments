# Experiment 7: 24-hour distributed-fitting quality/efficiency comparison

This is the approved long follow-up to the [short engineering screen](SHORT_TEST.md).
Experiment 5 is the saved control: `sdcfr5-par8-20261002-102757` by default.
It is **not retrained**. Experiment 7 uses eight local Ray actors for both
traversal and synchronous advantage-network fitting. This is an experimental
backend, **not a near-identical-output replacement or proven quality gain**.

## Frozen configuration

- Three seeds (0, 1, 2), each on a separate **n2-standard-16** VM, concurrently.
- **24 active hours per seed**; playable checkpoints at 6, 12, 18 and 24 hours,
  at the first completed outer iteration crossing each threshold.
- Exact Experiment 5 learner: 320 traversals per player/iteration, 200 advantage
  updates, global batch 2,048, five-million-row advantage reservoirs per player,
  structured lossless canonical inputs, 8 x 32 residual/LayerNorm networks,
  constant Adam learning rate 0.004, warm-start weights **and** Adam history,
  global-minibatch target standardisation, uniform historical-strategy mixture.
- Eight traversal/fitting workers per VM, one compute thread per worker.
  Only execution changes: `distributed_fitting=True`.
- Checkpoint serialization, reload validation and upload are excluded from the
  training clock. Traversal, fitting, startup and archive capture are included.
- All playable historical-policy archives, metadata and analysis retained.
  **No full replay/optimizer training-state dumps** (same retention as Experiment 5).

Each seed uses one VM, not eight VMs. The central replay retains the original
insertion and sampling rule. This tests the paper's synchronous distributed-gradient
principle, not its distributed replay or larger hyperparameters.

## Motivation and correctness

Cloud short run `sdcfr7-short-20261002-233043` passed single-update correctness.
Reported speedups: approximately **1.087x** for fitting, **1.068x** for short
matched-work end-to-end timing, and **1.061x** for observed nodes/second.
These are early-loop, one-source-seed measurements with roughly 60,619/33,336
replay rows, not mature five-million-row buffers or evidence of playing strength.

Accumulated fits were **not output-equivalent** (worst probed action-probability
difference 1.0). Float32 multiplication/reduction order can amplify tiny differences
through Adam, ReLU and regret matching. This study measures both quality and speed.

Cloud smoke retains these hard gates:

1. Real eight-worker single-step gradient/Adam comparisons for both players,
   including tiny, uneven and production-sized batches and nonempty Adam history.
2. Global target normalisation, identical sampling RNG, immutable frozen replay.
3. Non-finite values and failed worker synchronisation fail closed. Replica weights
   and Adam states must agree; failed partial fits are not committed or retried.
4. Three alternating-order 200-update frozen-fit comparisons **per player**, with
   identical initial weights, Adam history and batch streams.
5. Tiny training, full archive reload and evaluation for both backends.

Only accumulated full-fit near-output differences are permitted in the approved
comparative workflow, via explicit `--allow-trajectory-drift`. Tolerances are
unchanged. `full_fit_equivalence_passed` and legacy `passed` remain **false**
when drift is detected; `approved_for_comparative_run` separately reports whether
strict correctness checks permit the quality experiment. Without that flag,
the benchmark still enforces full-fit equivalence.

Fitting timing includes preparation, transfer, reduction and state synchronisation;
startup is separate. Production node throughput must be measured independently.
No training algorithm hyperparameters are adjusted to obtain a speedup.

## Evaluation against the existing Experiment 5

Reference metadata are checked before long training: learner configuration,
inputs, machine class, seeds, completed schedule, clock and node definitions must
match. Full archive hashes are checked before evaluation. Reference objects remain
read-only. Source commits and Python/NumPy/PyTorch versions are recorded: this is a
historical-control comparison, not contemporaneous randomised training.

- Primary quality endpoint: **24h Experiment 7 minus Experiment 5 two-seat EV**.
  The three same-seed pairs are the inferential units.
- All **nine cross-seed pairings at each checkpoint**, each with 50,000 duplicate
  deals (both seats; 100,000 hands). All-cell averages are descriptive: nine
  correlated cells are not nine independent replicates.
- Secondary endpoints: earlier paired EV, matched-active-time nodes and nodes/sec.
  Tables retain actual checkpoint overshoot as well as nominal checkpoint times.
- Shared duplicate-hand evaluator, split chance/action seeds, common random numbers.
  Uniform historical networks are sampled per player per hand; no expensive full
  behavioural-mixture reconstruction is needed for play.
- Usual rule-agent matches (10,000 duplicate deals per agent/seed/checkpoint) and
  later-versus-earlier matches (50,000 per pair/seed).
- **Routine LBR off** (`EVAL_LBR=0`). Complete playable archives remain available
  for later exploiter analysis. `EVAL_LBR=1` is an explicit cost-profiled addition.

Default workload: **114 tasks, 3.3 million duplicate pairs, 6.6 million hands**,
including 36 cross-experiment tasks. A CPU cost profile gates evaluation against
`EVAL_MAX_HOURS` (default 36). Matching completed shards are resumable.

Tables use pointwise training-seed t intervals (n=3), exploratory and not
multiplicity-adjusted. Head-to-head strength or higher throughput cannot establish
lower exploitability or Nash convergence. No model is automatically promoted.

## Retained outputs

Reuse `workers/`, `analysis/`, `evaluation_no_lbr/` (or `evaluation/` with LBR).
New artifacts under the evaluation directory:

- `exp7_vs_exp5_by_pair.csv`: raw cell results and finite-hand uncertainty.
- `exp7_vs_exp5_paired_head_to_head.csv`: paired summaries and training-seed intervals.
- `exp7_vs_exp5_cross_seed_descriptive.csv`: descriptive nine-cell averages.
- `exp7_vs_exp5_throughput_by_seed.csv` and `*_aggregate.csv`.
- `exp7_vs_exp5_head_to_head.png`, `exp7_vs_exp5_nodes_by_training_time.png`.
- `comparison_summary.json`, `comparison_interpretation.txt`,
  `reference_checkpoint_index.csv` and provenance in `evaluation_manifest.json`.

Training telemetry includes cumulative fitting/preparation time and the last
phase's actor compute/communication times. Actor times overlap; do not add them
to derive elapsed time. Central RSS excludes Ray/actor memory.
`smoke/fitting_benchmark/` retains correctness, drift and timing reports.

## Launch after committing and pushing

With `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` set for this repository:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr7-distfit-$(date -u '+%Y%m%d-%H%M%S')"
export EXP5_RUN_ID="sdcfr5-par8-20261002-102757"
export EVAL_LBR=0
export EVAL_PROFILE_ONLY=0
export EVAL_MAX_HOURS=36
unset RESUME_RUN_ID ADDITIONAL_HOURS
bash gcp/run_exp7_sd_cfr_distributed_fitting_24h.sh run
```

The commit must contain this update and be pushed. The launcher rejects older
refs without the comparison implementation. Use a new RUN_ID, not the short
screen's ID. No cloud resources are started by merely installing this code.

Stages: reference metadata/smoke → three parallel 24h runs → aggregation →
evaluation cost profile → evaluation. The baseline learner is trained only for
a tiny smoke fixture, never a new 24h control. Allow more than 24 elapsed hours
for provisioning, checks, checkpoint overhead and evaluation. Concurrent training
needs **48 N2 vCPUs** (three 16-vCPU VMs), plus the small controller.
Profile/evaluation use eight-vCPU CPU VMs.

To retry only evaluation, keep RUN_ID and EXP5_RUN_ID unchanged and use
`bash gcp/run_exp7_sd_cfr_distributed_fitting_24h.sh evaluate-only`.
This profiles again and reuses matching evaluation shards, without retraining.

For local validation with repository dependencies:
`bash gcp/run_exp7_sd_cfr_distributed_fitting_24h.sh smoke-local`.
Local ARM timings are not predictions of cloud N2 performance.
