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
- Each checkpoint: **1,000 LBR duplicate pairs**, 100 shards of 10 pairs,
  **4,096** preflop rollout samples. LBR is NOT exact exploitability.
- Every earlier/later checkpoint pair within each seed: **50,000 duplicate
  pairs** (18 temporal matchups).

For sampled play, a historical network is drawn separately for each player
once per hand, with a distinct RNG stream; it remains fixed throughout the
hand. LBR must not observe that hidden draw. Its hypothetical-hand queries use
the exact own-reach-weighted behavioural mixture, vectorised over bounded
model batches. There is no full-game-tree reconstruction or silent sampled
approximation to this queried mixture.

After training, a real-archive cost profile measures rule, LBR and temporal
play using the final archive of each seed. An extrapolation using the
worst measured per-pair cost and a 2x margin must fit **EVAL_MAX_HOURS=36** on
eight evaluation workers before full evaluation is launched. This is a guard,
not a guaranteed runtime. Profile has its own four-hour ceiling. If it fails,
training/aggregate outputs remain safe; inspect the profile before approving
a larger evaluation budget (up to 96 hours). No policy truncation is used to
force the evaluation to fit.

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

Under `gs://BUCKET/RUN_ID/`:

- `workers/task_.../archive/`: all playable policies, shared history chunks.
- Per-worker run/checkpoint manifests, `training_trajectory.csv`,
  `advantage_losses.csv`, `solver_diagnostics.csv`, completion/failure records.
- `analysis/`: training summary, checkpoint index, throughput table/chart.
- `evaluation/`: evaluator provenance, cost profile, resumable task results,
  per-seed and aggregate rule/LBR/temporal tables, quality-versus-time/node charts
  and temporal head-to-head charts. No cross-algorithm comparison outputs.
- `smoke/`: correctness, short evaluation and capacity reports. Synthetic
  full-capacity replay/archive stress data are deleted, not uploaded.

For analysis-only downloads (no policy archives):

```bash
RESULTS_BUCKET="${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation"
gcloud storage rsync --recursive "gs://$RESULTS_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='(^|/)(tasks|profile_tasks)(/|$)' \
  "gs://$RESULTS_BUCKET/$RUN_ID/evaluation" "cloud_outputs/$RUN_ID/evaluation"
```
