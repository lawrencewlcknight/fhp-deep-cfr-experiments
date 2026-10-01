# Experiment 1: SD-CFR implementation equivalence and efficiency

This number belongs to the **FHP Deep CFR repository**, not the FHP UCV repository.
It is a short fixed-work implementation test, not a new algorithm or strength study.
The reference is the current FHP SD-CFR port (already using compact float32 replay).
It is not the older Python-object replay implementation.

This audit was initially numbered Experiment 2 and was renumbered before cloud
submission. The former conventional Deep CFR Experiment 1 is preserved under
`archive_exp1_deep_cfr_best_config_transfer`. Historical outputs are not renamed.

## Arms and controls

1. `reference`: unchanged `SingleDeepCFRSolver`.
2. `scripted`: live scripted traversal inference; unchanged float32 replay.
3. `scripted_packed`: the same inference optimisation plus lossless bit-packed
   binary information-state features. Unpack minibatches to the original float32
   inputs; regret targets remain float32 and iteration indices int32.

Optimisation code is opt-in and does not change production defaults. It keeps
Python/NumPy RNG consumption, traversal order, alternating player updates,
continuous Adam state, diagnostics and every historical advantage snapshot.
Unsupported compilation/configurations or non-binary features raise errors.
No NumPy replacement sampler, target/gradient precision reduction, archive thinning,
extra traversal workers, or changed diagnostic schedule is included.

The small benchmark uses **three seeds (1234, 2025, 31415), three timing repeats**,
and **six complete CFR iterations per arm**. The first complete iteration is also
reported separately as warm-up. Each arm uses the same 320 traversals per player,
200 advantage updates, minibatch 2048, residual LayerNorm 8x32 network,
learning rate 0.004, standardised targets, uniform output mixture and no average
policy fitting. The replay capacity is reduced equally to **100,000/player** to
keep this screen inexpensive. The diagnostic interval stays at 25 (there is also
the existing final diagnostic). A smoke preset uses a 16-row reservoir to exercise
reservoir replacement, three iterations, four traversals and two updates.

Each fit runs in a **fresh process, sequentially on one machine**, with one Torch
intra-op and one inter-op thread. Arm order rotates across seeds/repeats; three
repeats place every arm in every order position for each seed. Timing repeats are
not additional independent seeds. Do not run another workload on the benchmark VM.

## Acceptance and measurements

Validate all historical network weights, final optimiser state, replay content
and counts, per-iteration node counts, losses, non-timing diagnostics, RNG states,
and probabilities/raw advantages on fixed independently sampled FHP observations.
Also compare 16 complete hands using the deployed per-hand network sampler and
verify normal archive save/reload. This never enumerates the FHP game tree.

Report both exact numerical equality and near equality (`atol=1e-6`, `rtol=1e-5`).
Replay features/indices, RNG states and node counts must match **exactly**;
only floating-point weights/targets/losses/probabilities admit tolerance.
Non-finite learner weights, losses or targets fail. Any failed comparison gives
a nonzero exit status after saving reports. A large difference is a bug/regression
to investigate, not evidence against SD-CFR itself.

Timings include traversal, fitting, archive capture and unchanged diagnostics.
Report total training, collection/fitting/archive components, post-first-iteration
time, nodes/second, and **initialisation plus training** (including compilation).
Policy probes, comparison and disk serialization happen after training and are
excluded. Report paired time ratios and percentages, per-seed medians and repeat
ranges. A ratio below one means slower; no speedup is assumed or guaranteed.

Memory reports distinguish allocated replay-array bytes, populated rows and peak
training-process RSS. The five-million-row projection is **array arithmetic**, not
a measured full-capacity benchmark. Short runs cannot establish long-horizon
speedup, mature-buffer performance, archive scaling or policy strength. A later
longer/full-reservoir test is needed before production promotion.

## Local run

From this repository with its dependencies installed:

```bash
python -m experiments.fhp.exp1_sd_cfr_efficiency.run --smoke
python -m experiments.fhp.exp1_sd_cfr_efficiency.run
```

For a smaller pilot (not the full three-seed result):

```bash
python -m experiments.fhp.exp1_sd_cfr_efficiency.run --seeds 1234 --repeats 1
```

Use `--output-dir PATH` for a new output directory. Existing directories are rejected
to avoid mixing results. `--threads N` applies equally to all arms; do not compare
outputs produced under different thread settings as a paired equivalence test.

## GCP run, after committing and pushing

Assume `PROJECT_ID`, `REGION`, `BUCKET` (including `gs://`) and `SA_EMAIL` are set.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="fhp-sdcfr-exp1-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp1_sd_cfr_efficiency.sh run
```

One `n2-standard-8` VM runs the smoke first, then the full benchmark only if it
passes. No controller or parallel child jobs are needed. The two-hour job cap
includes dependency setup; this is a short benchmark rather than a 24-hour run.
The shared launcher accepts a pinned commit SHA as well as a branch/tag.

```bash
gcloud batch jobs describe "$RUN_ID" --project "$PROJECT_ID" --location "$REGION"
mkdir -p "outputs/downloaded/$RUN_ID"
gcloud storage rsync --recursive "$BUCKET/$RUN_ID/outputs/$RUN_ID" "outputs/downloaded/$RUN_ID"
```

The uploader preserves the `outputs/` directory under `$BUCKET/$RUN_ID/`, hence
the path above. Keep the chosen run ID for later download.

## Retained outputs

`manifest.json` records source hashes (including uncommitted local code), commit,
versions, machine, configuration and tolerances. `schedule.json`, individual logs
and run records, `runs.json`, `comparisons.json`, `summary.json`, `report.md` and
`speed_and_replay_memory.png` contain the audit and timings.

Large replay/optimiser validation payloads and temporary playable archives are
deleted after each comparison block. Only compact content fingerprints remain.
This benchmark does not retain resumable training states or promote a default.
