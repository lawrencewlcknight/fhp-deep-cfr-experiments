# Experiment 2: optimised uniform SD-CFR, 24-hour FHP benchmark

## Frozen configuration

This transfers the selected uniform SD-CFR configuration described in
`docs/SD_CFR.md`, using Experiment 1's scripted-live-inference plus packed
binary-replay implementation. It matches the time, hardware, game and policy
evaluation protocol of FHP UCV-ESCHER Experiment 1, not its neural architecture.

- Exactly three training seeds: **0, 1, 2**.
- One separate on-demand **n2-standard-8** VM per seed; all three concurrently.
  This requires 24 N2 vCPUs plus a small controller VM.
- **24 active training hours**, stopping at a complete outer iteration.
  A 1,000,000-iteration safety cap is not the stopping target; hitting it early
  fails the completeness checks. The cloud training ceiling is 36 wall hours.
- Policies at the first complete iteration crossing **6, 12, 18, 24 hours**.
- Raw canonical OpenSpiel FHP information-state features, with no new abstraction.
- Residual/layer-normalised centred-advantage **8 x 32** networks.
- **320** external-sampling traversals per player/iteration; **200** advantage
  updates per player/iteration; minibatches **2,048**; Adam **0.004**, constant.
- Continuous warm start including persistent Adam state, existing target
  standardisation and iteration-weighted advantage regression.
- **5,000,000** uniformly sampled replay rows per player. Binary features are
  losslessly packed; sampled inputs and targets are float32.
- Uniform historical-strategy weights, capturing every post-player-update
  network. No output-policy fitting, archive thinning, or final-network proxy.
- One Torch intra-op/inter-op thread per training process, as in the optimised
  implementation audit. Same VM allocation as UCV, not identical thread usage.

Checkpoint serialization, reload validation and upload pause the active clock.
Traversal, optimisation, normal diagnostics and regular historical capture
remain charged. Actual crossing time, elapsed time, excluded overhead and
nodes touched are recorded. UCV's clock additionally excluded average-policy
fitting: this is not an equal end-to-end deployment-cost test.

## Retention and failure handling

Each seed writes immutable float32 NumPy chunks of at most 128 complete
iterations. Four small manifests reference archive prefixes; historical
weights are saved once, not copied into four cumulative archives. Storage
still grows linearly with iterations: these weights ARE the SD-CFR policy.
The archive is streamed to disk instead of retaining all historical networks
in RAM. Checksums and contiguous-prefix checks protect reloading.

Retain all policy chunks/manifests, analysis, logs and metadata. **No full
training replay or optimiser dumps are retained.** Thus checkpoints support
play and evaluation, not resumption of learning. No automatic training retry
is enabled: silently restarting a 24-hour trajectory would double cost and
could mix checkpoints. Failed partial runs remain available for diagnosis;
restart training only with a new run ID. Evaluation tasks, by contrast, are
individually checkpointed and can resume without retraining.

## Standalone evaluation

This experiment trains and evaluates SD-CFR only. **No UCV run IDs,
checkpoints, completed evaluation results or access to the Escher bucket are
required.** BUCKET is the SD-CFR output bucket. Old UCV_EXP1_RUN_ID and
UCV_EVAL_RUN_ID terminal exports are ignored. Cloud smoke evaluates only the
short SD-CFR training run it creates itself.

Cross-algorithm comparisons will be performed separately later; no comparator
tables are imported and no SD-CFR-versus-UCV games are scheduled here. To support
that later analysis, the five corrected rule agents, game, both-seat duplicate
protocol, base seed 20260922, and split chance/action-seed layout remain unchanged.

- Each of 12 SD-CFR checkpoints: **10,000 duplicate-deal pairs per rule agent**.
- **LBR is omitted by default for Experiments 2–5** (`EVAL_LBR=0`).
  Opting in with `EVAL_LBR=1` adds **1,000 LBR duplicate pairs** per checkpoint,
  100 shards of 10 pairs and **4,096** preflop rollout samples.
  LBR is NOT exact exploitability. Experiment 6 retains its prior default.
- Every earlier/later checkpoint pair within each seed: **50,000 duplicate
  pairs** (18 temporal matchups).

For sampled play, a historical network is drawn separately for each player
once per hand, with a distinct RNG stream; it remains fixed throughout the
hand. LBR must not observe that hidden draw. Its hypothetical-hand queries use
the exact own-reach-weighted behavioural mixture, vectorised over bounded
model batches. There is no full-game-tree reconstruction or silent sampled
approximation to this queried mixture.

After training, a real-archive cost profile measures rule and temporal
play using the final archive of each seed. An extrapolation using the
worst measured per-pair cost and a 2x margin must fit **EVAL_MAX_HOURS=36** on
eight evaluation workers before full evaluation is launched. This is a guard,
not a guaranteed runtime. Profile has its own four-hour ceiling. If it fails,
training/aggregate outputs remain safe; inspect the profile before approving
a larger evaluation budget (up to 96 hours). No policy truncation is used to
force the evaluation to fit. With `EVAL_LBR=1`, the profile also measures LBR
and requires the full-archive numerical validation gates. The no-LBR path
never constructs the expensive behavioural-mixture evaluator or runs LBR probes.

Uncertainty for quality curves is across three independent training seeds.
Temporal matchups pair later and earlier policies within the same training seed.
Node charts connect observed checkpoint means; no exact exploitability curve or
convergence certificate is claimed. Playable checkpoints, evaluator provenance
and per-seed results are retained for the later cross-algorithm analysis.

## Launch after committing and pushing

From this repository, with PROJECT_ID, REGION, BUCKET and SA_EMAIL configured:

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr2-24h-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp2_sd_cfr_24h.sh run
```

BUCKET accepts either a bucket name or `gs://...`. The launcher rejects a
stale revision without Experiment 2. The service account needs the established
Batch worker/storage/logging permissions plus `roles/batch.jobsEditor` and
permission to act as itself (`roles/iam.serviceAccountUser`) for child jobs.
The laptop may disconnect after the remote controller is submitted.

Pipeline: cloud functional/equivalence/mature-storage smoke -> three training
VMs -> training aggregation -> evaluation cost profile -> full evaluation and
standalone charts. No cloud resources are launched by local tests.

```bash
bash gcp/run_exp2_sd_cfr_24h.sh status
# Inspect generated job definitions without submitting:
bash gcp/run_exp2_sd_cfr_24h.sh dry-run
# Local reduced training + evaluation smoke, using the active Python environment:
bash gcp/run_exp2_sd_cfr_24h.sh smoke-local
```

To repeat the profile and resume evaluation ONLY, keep the same RUN_ID and
set REPO_REF to the intended pushed evaluation code:

```bash
# Increase only after reviewing the measured evaluation_profile.json:
# export EVAL_MAX_HOURS=48
bash gcp/run_exp2_sd_cfr_24h.sh evaluate-only
```

Existing completed task shards are reused only when checkpoint hashes and
the evaluation protocol and implementation hashes match. Different versions
use separate shard namespaces, preserving the previous results without mixing
them into a rerun. Input paths may change across VMs. Profile
outputs, partial evaluation tasks and failures are retained; production
training is never repeated by `evaluate-only`.

## Outputs

### Complete the existing Experiments 2–5 without LBR (no retraining)

All three training workers and aggregation completed in each run below. The
failed stage was the old LBR cost profile. **Keep these original RUN_IDs** and
use `evaluate-only`, not `run`. Once these changes are committed and pushed:

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export EVAL_LBR=0
export EVAL_LBR_DEVICE=cpu
export EVAL_WORKERS=8
export EVAL_MAX_HOURS=12
export EVAL_PROFILE_ONLY=0

export RUN_ID="sdcfr2-24h-20261002-003338"
bash gcp/run_exp2_sd_cfr_24h.sh evaluate-only

export RUN_ID="sdcfr3-24h-20261002-010643"
bash gcp/run_exp3_sd_cfr_structured_24h.sh evaluate-only

export RUN_ID="sdcfr4-vm16-20261002-095614"
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh evaluate-only

export RUN_ID="sdcfr5-par8-20261002-102757"
bash gcp/run_exp5_sd_cfr_parallel_24h.sh evaluate-only
```

Use the SD-CFR `BUCKET`, project, region and service account. Each command
submits a remote controller and returns; each controller runs a short CPU
profile followed by evaluation on one `n2-standard-8` VM. Four concurrent
evaluations require 32 N2 vCPUs plus four small controllers. No training or
cross-algorithm matches are launched; the laptop may disconnect.

Each experiment still evaluates all 12 saved policies (three seeds at
6/12/18/24 hours): 60 rule-agent tasks with 10,000 duplicate pairs each and
18 within-seed temporal tasks with 50,000 pairs each. This totals **1.5 million
duplicate pairs (3 million hands)** per experiment, with unchanged opponents,
deal seeds and sampling protocol. Reports use training seeds as the inferential
unit. These matches assess playing strength against the tested opponents,
not exploitability or Nash convergence.

Outputs go to **`$BUCKET/$RUN_ID/evaluation_no_lbr/`**, separate from the old
LBR profile/results under `evaluation/`. The manifest marks LBR as deliberately
omitted; charts do not show an empty/zero LBR series. All source policy archives
and manifests under `workers/` remain untouched, including every historical
network needed for a later exact-mixture LBR or other exploiter assessment.
Old failed Batch job records are not changed by successful recovery jobs.

The original CPU profiles measured rule/temporal costs implying about
3.02/3.06/3.14/2.64 hours of eight-worker computation for Experiments 2/3/4/5,
respectively. Applying their 2x safety margin gives about 6.05/6.12/6.28/5.27
hours. **Allow roughly 6–8 elapsed hours per experiment** including ordinary
setup/transfer overhead, excluding quota/capacity waits; this is an estimate,
not a measured completed run. The new pilot checks these estimates again.
`EVAL_MAX_HOURS=12` is a safety allowance, not the expected runtime.

For a local smoke with an existing short training output:

```bash
python -m experiments.fhp.exp2_sd_cfr_24h.evaluate smoke \
  --source "$SMOKE_SOURCE" --output "$SMOKE_SOURCE/evaluation_no_lbr" --workers 2 --skip-lbr
```

The Python evaluator keeps LBR available explicitly; cloud launchers 2–5 and
their `smoke-local` actions default to omitting it. Setting `EVAL_LBR=1` opts
back into LBR, including its potentially expensive profile. A stale GPU device
setting is ignored when LBR is disabled. No no-LBR profile can authorize an
LBR evaluation because the task fingerprints differ.

### LBR efficiency repair (Experiments 2–5; shared path also used by 6)

The October 2026 failures occurred in the **evaluation cost profile**, after
all training seeds and training aggregation succeeded. The scalar LBR queried
roughly 1,000 hypothetical opponent hands separately. Every query reconstructed
own reach by repeatedly scanning the entire historical-network archive, including
earlier decisions already queried for the same hand. The final policies have
roughly 12,000 networks per player, unlike the single deployed policy network
used by UCV-ESCHER and VR-Deep. Increasing the timeout is not a sufficient fix.

`deep_cfr_poker/sd_cfr_lbr.py` now batches across hypothetical hands and networks,
deduplicates identical float32 inputs, and computes the range likelihood directly
as the mean historical own reach. For uniform historical weights, the product of
conditional mixture probabilities along the player's own history telescopes to
that mean. The conditional next-action distribution still weights each network
by its own reach: this is **not** the pointwise average of network predictions.
Raise-response probabilities are evaluated in the same pass; queries with no
legal raise compute only the required reach. Caches and model batches are bounded.

All historical networks, legal opponent hands, 4,096 preflop rollouts, exact flop
equity, action scoring and tie-breaking remain unchanged. LBR inherits the shared
scorer/equity/RNG implementation. The responder never sees the historical-network
identity sampled secretly for actual SD-CFR play. No retraining, network thinning,
policy distillation or lower-precision weights are used.

The common UCV retrospective protocol remains 1,000 duplicate deal pairs per
checkpoint, 100 shards of ten, both seats, base seed 20260922, deal seed
base+1000000+shard, rollout seed base+1500000, split chance/action seed arrays,
and milli-big-blinds per hand. The shared `lbr.py`, `cards.py`, `equity.py` and
`game.py` are unchanged from the suite used by UCV and VR-Deep. **The generic
VR-Deep adapter does not by itself prescribe this cohort/shard/seed schedule**;
use this same protocol when producing its comparative results. LBR remains a
sampled best-response lower-bound diagnostic, not exact exploitability.

Local real-weight diagnostics used only the first 128 networks from Experiment 2
seed 0 and Experiment 3 seed 0. On five fixed preflop/flop information states,
target-query speedups were 9.6–13.0x and 41.5–65.5x, respectively; maximum
normalised range error was below 3.1e-9, fold-probability error below 1.8e-8, and
all tested LBR actions agreed. These are one-repeat, CPU **query** timings,
not full-evaluation speedups or guarantees about the complete trained policies.
The reproducible audit is `python -m deep_cfr_poker.sd_cfr_lbr_audit --help`;
measurements are in `benchmarks/sd_cfr_lbr_cpu_20261003.json`.

The full-archive profile now runs a numerical gate against the original scalar
mixture on every final training seed. Missing/failed gates stop evaluation.
The existing two-times-margin cost guard is retained. Source hashes and explicit
CPU/CUDA settings isolate old/new task namespaces; results cannot be silently
mixed. Production evaluation still requires a matching passed profile.

#### Evaluation-only GPU pilot (opt-in; not yet hardware-validated)

CPU remains the default. An optional CUDA path performs the same float32 neural
inference, with TF32/mixed precision disabled and float64 CPU mixture accumulation.
It uses a bounded GPU weight cache. The pilot uses one `g2-standard-8` VM with
one L4 and two evaluation processes by default; training VMs and training code
are unchanged. Batch installs GPU drivers, the evaluation VM installs the CUDA
build of the same PyTorch 2.7.0 release, and CPU/GPU equivalence tests run before
the full-archive profile. A requested but unavailable GPU fails explicitly.
GPU quota/capacity is required; no CPU fallback or automatic budget escalation.

After pushing the implementation, retain an **existing completed training RUN_ID**
and the experiment's corresponding launcher. For example, an Experiment 2 pilot:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr2-24h-20261002-003338"
export EVAL_LBR=1
export EVAL_LBR_DEVICE=cuda
export EVAL_WORKERS=2
export EVAL_MAX_HOURS=36
bash gcp/run_exp2_sd_cfr_24h.sh profile-only
```

`profile-only` launches neither training nor full evaluation. It validates/times
the existing archives and saves `evaluation/evaluation_profile.json`. Review its
numerical checks, measured runtime and cost estimate before approving evaluation.
If acceptable, `evaluate-only` re-runs the gate/profile, then evaluates those same
saved policies. It does not retrain. Use the corresponding Experiment 3–5 launcher
and original RUN_ID for those policies. Set `EVAL_LBR_DEVICE=cpu` and
`EVAL_WORKERS=8` for the CPU path. Do not claim a two-hour completion time until a
representative full-archive pilot has measured it.

The GPU job shape follows Google's Batch GPU documentation:
https://docs.cloud.google.com/batch/docs/create-run-job-gpus
and the evaluation wheel pairing follows:
https://pytorch.org/get-started/previous-versions/#v270

### Stored artefacts

Under `gs://BUCKET/RUN_ID/`:

- `workers/task_.../archive/`: all playable policies, shared history chunks.
- Per-worker run/checkpoint manifests, `training_trajectory.csv`,
  `advantage_losses.csv`, `solver_diagnostics.csv`, completion/failure records.
- `analysis/`: training summary, checkpoint index, throughput table/chart.
- `evaluation/`: evaluator provenance, cost profile, resumable task results,
  per-seed and aggregate rule/LBR/temporal tables, quality-versus-time/node charts
  and temporal head-to-head charts, when LBR is enabled.
- `evaluation_no_lbr/`: the routine Experiments 2–5 evaluation, with rule-agent
  and temporal results only and explicit LBR-omission metadata. No
  cross-algorithm comparison outputs or source-checkpoint deletions.
- `smoke/`: correctness, short evaluation and capacity reports. Synthetic
  full-capacity replay/archive stress data are deleted, not uploaded.

For analysis-only downloads (no policy archives):

```bash
RESULTS_BUCKET="${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation_no_lbr"
gcloud storage rsync --recursive "gs://$RESULTS_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='(^|/)(tasks|profile_tasks)(/|$)' \
  "gs://$RESULTS_BUCKET/$RUN_ID/evaluation_no_lbr" "cloud_outputs/$RUN_ID/evaluation_no_lbr"
```
