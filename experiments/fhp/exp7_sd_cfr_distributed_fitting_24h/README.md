# Experiment 7: synchronous distributed fitting, 24 hours

Experiment 5 is the unchanged control. Experiment 7 reuses its eight local Ray
actors for both traversal and advantage-network fitting. This is an
**experimental execution backend, not a demonstrated speed improvement**.

**Current validation status (2026-10-02): do not treat this as a validated
drop-in optimization.** The real-worker single-update checks pass, but the
strict full-fit output-equivalence benchmark fails locally. Keep the preflight
gate; do not launch a long run without reviewing the evidence below.

For the agreed **standalone short screen**, use
[the 30-minute test](SHORT_TEST.md). It runs only one VM, reports accumulated
full-fit drift without automatically promoting the learner, and never starts
this 24-hour workflow. Single-update correctness remains a hard gate.

## Frozen scientific configuration

- Three seeds (0, 1, 2), each on its own n2-standard-16 VM.
- 24 active training hours; playable checkpoints at 6, 12, 18 and 24 hours.
- 320 traversals per player per iteration, divided across eight actors.
- 200 optimizer updates per player, global batch 2048 (normally 256 per actor).
- Identical structured observable inputs, 8 x 32 residual/LayerNorm networks,
  constant Adam learning rate 0.004, warm-start weights AND Adam state,
  five-million-row central advantage reservoirs, target standardization and
  uniform historical-strategy output mixture.
- Identical standalone rule-agent, LBR and temporal head-to-head evaluation.
- Playable historical policies, analysis and metadata retained; no full
  training-state dumps (this follows Experiment 5, not resumable Experiment 6).

## Distributed fitting

The central reservoir retains the original insertion and sampling algorithms.
For each fit, it draws exactly the same 200 global minibatches as the baseline,
standardizes targets over each whole minibatch, and shards its rows among the
eight actors. Features remain losslessly packed during transfer. Each actor
unpacks only its current minibatch. No actor owns a second full replay buffer.

Persistent Gloo processes compute local backward passes, sum example-weighted
gradients, and then make identical Adam updates. There are no asynchronous or
independent local optimization steps. Uneven and empty shards use the same
global loss denominator. Workers receive one RPC per full fit, with direct
Gloo collectives between optimizer steps. Replicas must have identical final
weights AND Adam moments before rank 0's state is accepted by the driver.
The next player then traverses against the newly fitted strategy.

This follows the paper's synchronous distributed-gradient principle, not its
40-million-row distributed replay or larger training hyperparameters. It uses
one VM per seed, not an eight-VM cluster. The original Experiment 5 classes and
defaults remain unchanged.

Float32 batch-matrix multiplication and gradient-reduction order differ from
the central implementation. The objective and global batch remain unchanged,
but bitwise-identical fits or long stochastic trajectories are not promised.
Small numerical differences can amplify through repeated Adam/ReLU updates
and regret matching. Output quality must therefore be evaluated as well as speed.

## Validation and speed measurement

Cloud smoke checks:

1. Exact global replay batches, global target normalization and sampling RNG.
2. Real eight-actor gradients and Adam steps versus the central reference,
   including batch sizes smaller than the worker count and uneven shards.
3. Actor failures fail closed; no partial fit is committed or silently retried.
4. A three-repeat, production-sized 200-update frozen-fit benchmark. Both
   arms begin with identical weights, nonempty Adam state and sampling seeds.
   Arm order alternates. The source replay is generated once with a frozen
   policy and independent traversal-phase seeds, then retained unchanged.
5. Actual training, archive reload, and shared evaluator smoke.

The benchmark reports startup separately and includes replay sampling,
preparation, transfer, all-reduce and state synchronization in fitting time.
It is NOT a full-run speedup estimate. The fixed global batch's numerical
comparison is separate from all-replica synchronization checks.

The full-fit equivalence gate is deliberately conservative: parameter/Adam and
logit tolerances are absolute 0.0002 and relative 0.002; the maximum probed
action-probability difference must be <= 0.002. A failed gate retains timing
and difference reports and **stops the controller before paid long training**.
Do not bypass it without explicitly reviewing the numerical drift and agreeing
that the new arm will be assessed as a statistically comparable learner, not a
near-identical-output execution replacement.

Small networks can make eight-way fitting slower because collectives, Adam
replication and data preparation outweigh parallel matrix multiplication.
Same-VM production nodes/hour is the primary efficiency measure; more nodes
are not themselves proof of better policy quality.

## Outputs

The baseline workers/, analysis/ and evaluation/ schemas are reused. Additional
trajectory columns record cumulative fitting time, preparation time, updates
and examples, and the final phase's mean worker compute/communication time.
Actor times overlap and MUST NOT be added to derive elapsed time. Central RSS
still excludes Ray and actor memory, as explicitly stated in the manifest.
The smoke's fitting_benchmark/ contains per-repeat numerical comparisons,
raw timing observations and summary.json, even if full-fit equivalence fails.

## Initial local evidence (not cloud performance)

On macOS ARM64, PyTorch 2.7.0 / Ray 2.51.2, three alternating-order repeats of
200 updates at global batch 2048 gave median central fitting time **1.698 s**
and distributed fitting time **3.001 s** (0.566x throughput, not a speed-up).
Both include sampling and fit overhead; actor startup (17.50 s) is excluded.
This is one frozen source/seed with three optimizer sampling streams, not
three independent long training runs. The replay fixture had 2081 rows;
the production five-million-row replay was not memory-profiled by this check.

Direct all-reduced gradients and individual Adam updates passed tight numerical
checks for 3-, 17- and 2048-example batches, including nonempty Adam state.
All replicas agree exactly. However, accumulated 200-update fits failed the
full-fit near-output checks: in the first repeat, the largest probe logit
difference was about 0.398 and the largest action-probability difference was
1.0. Individual-update agreement does not justify calling long fits identical.
The central/distributed final regression losses in that repeat were about
0.7522/0.7537; neither those losses nor isolated policy-probe differences
establish playing strength.

The complete ordinary test suite passed (191 tests, 11 opt-in skips); all four
new real-Ray integration cases passed separately. Actual Experiment 7 smoke
training saved all four policy prefixes and the shared evaluation completed
all 13 smoke tasks. The separate full-fit gate correctly remains failed.
A short n2-standard-16 measurement is needed before extrapolating efficiency.

## Commands (after commit and push)

With the existing project, region, bucket and service-account variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr7-distfit-$(date -u '+%Y%m%d-%H%M%S')"
unset RESUME_RUN_ID ADDITIONAL_HOURS
bash gcp/run_exp7_sd_cfr_distributed_fitting_24h.sh run
```

This submits smoke, three training VMs, aggregation, evaluation cost profiling
and evaluation. The controller does not reach training if any smoke gate fails.
No cloud resources are started merely by installing or testing this code.

Local validation with the repository dependencies:

```bash
RUN_RAY_SD_CFR_TESTS=1 python3 -m pytest -q tests/test_sd_cfr_distributed.py tests/test_exp7_sd_cfr_distributed_fitting.py
python3 -m experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.benchmark --output /tmp/sdcfr7-fitting-check
```

Choose a fresh output directory each time. Do not compare local ARM timings
directly with n2-standard-16 results.
