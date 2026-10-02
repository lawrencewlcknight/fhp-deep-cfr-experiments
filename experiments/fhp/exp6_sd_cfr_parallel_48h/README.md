# Experiment 6: parallel structured SD-CFR, 48 active hours

This is the long-horizon counterpart of Experiment 5, analogous to UCV-ESCHER
Experiment 8. It does **not** introduce a new algorithm or input representation.
The only planned differences are the 48-hour horizon and a final resumable
training state. No cloud job is started by creating this experiment.

## Frozen configuration

- Seeds `0, 1, 2`, run concurrently on three separate `n2-standard-16` VMs
  (16 vCPUs, 64 GiB each; 48 N2 vCPUs total, plus the small controller).
- Eight Ray traversal actors per VM, one central learner, one Torch thread
  per process. Still **320 total traversals per player/iteration**, not 320
  per worker; the assignment is 40 per actor.
- Exact suit-canonical 183-feature observable inputs, losslessly packed
  two-bit replay, two 5-million-row advantage reservoirs.
- Same 8 x 32 residual LayerNorm centred advantage networks, 200 advantage
  updates per player/iteration, minibatches of 2,048, Adam learning rate 0.004,
  target standardisation and continuous warm-started networks/optimisers.
- Uniform historical-strategy mixture, no average-policy network fitting.
- 48 active hours per seed, with a one-million-iteration safety cap. Complete
  the current two-player iteration before stopping; consequently there can be
  a small budget overshoot. Ray startup, collection, fitting and routine archive
  capture count as training; policy-checkpoint I/O, validation and uploads do not.
- Playable checkpoints at the first completed iterations crossing
  **6, 12, 18, 24, 30, 36, 42 and 48 hours**. Historical networks are stored once
  in shared immutable chunks, never thinned or reduced in precision.

## Outputs and retention

Normal outputs match Experiment 5: manifests, node/time trajectories, losses,
solver diagnostics, playable policy archive prefixes, training summaries/charts,
sampled rule-agent scores, LBR and temporal head-to-head results. Evaluation is
standalone; cross-algorithm matchups remain a later analysis.

Every checkpoint uses the same five published rule agents (10,000 duplicate
deals each), LBR (1,000 deals, 4,096 preflop rollouts) and all later-versus-earlier
within-seed pairs (50,000 duplicate deals per pair). There are now 28 temporal
pairs per seed instead of six. LBR remains a sampled lower-bound diagnostic,
**not exact exploitability or proof of convergence**. A latest-checkpoint cost
pilot precedes evaluation and can stop it if the requested cost cap is too low;
this never discards the completed training outputs.

Only the final endpoint additionally contains `training_state/`:

- `learner.pt`: both advantage networks, Adam states, global iteration/node
  counters, normalisation/diagnostic state, Python/NumPy/Torch random states,
  and cumulative reporting context;
- six packed `.npy` replay arrays, including both reservoir stream counters;
- `manifest.json`: configuration, runtime/code fingerprints, checksums and a
  reference to the adjacent complete historical strategy archive.

The state is saved **after final solver diagnostics**, which may consume random
numbers. Its ready manifest is written last. There are no intermediate replay
dumps. At full capacity the two packed replays occupy approximately 620 MB
decimal in total per seed, plus learner/reporting state. The historical strategy
archive is additional and can be much larger; retaining all historical networks
is intrinsic to this SD-CFR output policy. The final state reuses that archive
without duplicating it within the run. Evaluation/aggregation VMs skip downloading
`training_state/` because playable policy archives suffice.

The state is CPU-only. Restore using the same implementation, Python, Torch,
NumPy, Ray and OpenSpiel versions and thread settings. Mismatched configuration,
runtime, code, missing history or damaged arrays fail closed. Only load trusted
states: `learner.pt` includes Python pickle data. Ray actors are reconstructed
from the saved execution recipe; their trajectory streams depend on run seed,
global iteration, player and worker, so actor processes need not be pickled.
Cloud continuation selects the original Python 3.11 patch version from the
saved manifest; the scientific dependency versions remain pinned in the repo.

## Start on GCP (after committing and pushing)

Use the usual FHP SD-CFR `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr6-48h-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=36
unset RESUME_RUN_ID ADDITIONAL_HOURS
bash gcp/run_exp6_sd_cfr_parallel_48h.sh run
```

The cloud controller runs smoke tests, three parallel training tasks, aggregation,
the cost pilot and evaluation. The smoke includes a real eight-actor save/reload/
continue equivalence test, a CLI continuation test, and playable-policy reload.
Each training task has a 72-hour wall-clock ceiling to allow checkpoint/upload
overhead beyond the 48 active hours. Total elapsed completion includes startup
and evaluation; 48 hours is not the total job duration. No local smoke is needed
before a normal cloud launch. A local smoke is available with `smoke-local`.

## Continue later without restarting learning

Use a **new** destination run ID and the original saved code revision. This
example continues all three final states from the nominal 48h endpoint to 72h:

```bash
export REPO_REF="<the original run's full commit SHA>"
export RESUME_RUN_ID="<the completed sdcfr6-48h run ID>"
export ADDITIONAL_HOURS=24
export RUN_ID="sdcfr6-resume-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp6_sd_cfr_parallel_48h.sh run
```

Continuation supports another 6..48 active hours in multiples of six. The
target is the old **nominal** endpoint plus this amount; any old iteration
overshoot is already counted. It restores replay, optimisers, RNG and the
complete historical policy mixture, preserves earlier playable checkpoints and
reporting rows, and appends new six-hour checkpoints plus one new final state.
Old outputs are not overwritten. The new run copies the previous archive so it
is self-contained; that does duplicate historical archive storage **across**
run IDs. Elapsed-time reporting excludes the gap between separate jobs and the
restore/download phase; active-time reporting remains cumulative.

For a local continuation retain the entire source worker's `archive/` and
`training_state/` directories together, not just the final JSON manifest:

```bash
python -m experiments.fhp.exp6_sd_cfr_parallel_48h.train \
  --seed 0 --output-root results/new_continuation \
  --resume-state results/original/workers/task_000_parallel_structured_uniform_sd_cfr_48h_seed_0/training_state/manifest.json \
  --additional-hours 24
```

The unit/integration tests compare saved/reloaded continuation with an in-memory
continuation at the **same stopping boundary**, checking exact networks, Adam,
replay replacements, RNGs, losses, nodes and full historical strategies. This is
not a claim that a wall-clock-stopped 48h+24h run is bit-identical to a fresh
uninterrupted 72h run: final-stop diagnostics consume RNG and runtime scheduling
can change the number of completed iterations at a time threshold.

## Verification

```bash
python -m pytest -q tests/test_sd_cfr_training_state.py tests/test_exp6_sd_cfr_parallel_48h.py
RUN_RAY_SD_CFR_TESTS=1 python -m pytest -q tests/test_exp6_sd_cfr_parallel_48h.py -k real_ray
```
