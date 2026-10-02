# Experiment 5: eight-worker structured SD-CFR, 24 hours

This is the parallel counterpart to **Experiment 4**. Both use the identical
`n2-standard-16` allocation, structured card representation, SD-CFR learning
configuration, training budget and evaluation protocol. Only traversal execution
changes. It is analogous to the sequential/parallel FHP UCV-ESCHER VM pair.

## Frozen comparison

| Setting | Experiment 4 | Experiment 5 |
| --- | --- | --- |
| Per-seed training VM | n2-standard-16, 16 vCPU, 64 GiB | Identical |
| Seeds / active time per seed | 0, 1, 2 / 24 hours | Identical |
| Traversal collection | Sequential | Eight local Ray actors |
| Total traversals/player/iteration | 320 | 320: 40 per actor |
| Central advantage optimisation | 200 updates, batch 2,048 | Identical |
| Playable policy checkpoints | 6, 12, 18, 24 active hours | Identical |
| Torch threads per process | 1 | 1 |
| Disk / provisioning | 200 GiB pd-balanced / STANDARD | Identical |

The learning configuration is imported from Experiment 4: the exact 183-feature
player-observable suit-canonical encoder, 8 x 32 residual/layer-normalised
advantage networks, Adam learning rate 0.004, continuous weights and optimiser,
target standardisation, five-million-row advantage replay per player and uniform
historical-policy weighting. There is no average-policy fit, critic or change
to what the policy observes. Lossless two-bit structured replay is retained.

The three training seeds run concurrently on **three separate VMs**. Within
each VM, one central learner manages eight traversal actors. This is not eight
VMs per seed, nor eight seeds: the training allocation remains **48 N2 vCPUs**
plus the small controller. Only traversal is parallel; gradient updates remain
centralised. Serial optimisation, replay insertion, actor communication and
archive capture will limit speedup, so an eightfold improvement is not assumed.
The unchanged auxiliary evaluation stage also uses eight workers, separately
from the eight training actors.

## Algorithm and execution safeguards

Each player phase freezes and broadcasts both advantage networks. All actors'
samples return before the learner updates that player's network, and the next
player sees the updated weights. Actor results enter the central reservoir in
worker-index/traversal order, independent of completion order. Actors do not
thin samples or maintain private sampling reservoirs. Packed float32 targets,
bounded transfer chunks, shared frozen snapshots and a phase-local exact
inference cache reduce overhead without changing the learning rule.

Random streams are deterministically derived from run seed, iteration, player
and worker. Therefore the same seed label does **not** imply the same sampled
trajectory as the original sequential collector. Validation compares eight
Ray actors against eight serial reference workers with identical streams,
checking replay contents, optimiser state, losses, historical networks and
node counts. The production-partition test uses 320 traversals split 40 each;
the short training/evaluation smoke uses eight traversals split one each.
Actor failures and malformed data fail closed rather than silently reducing
the traversal budget or duplicating samples.

The cloud smoke gates full training on real-Ray parity and actor failure
checks, an actual eight-actor training/checkpoint reload, the central
five-million-row replay/archive capacity stress, and sampled evaluation.
The capacity stress checks central storage, not a full-duration aggregate-RSS
measurement of the eight-actor process tree.

## Outputs and comparison

The shared baseline pipeline provides:

- `workers/`: run/configuration/encoder/execution manifests, per-iteration
  nodes and active time, losses, diagnostics, four playable policy prefixes
  and immutable historical-network chunks stored once.
- `analysis/`: checkpoint summary, node/time metrics and throughput chart.
- `evaluation/`: identical rule-agent matches, LBR diagnostics, temporal
  head-to-head matches and quality plots indexed by nodes and time.
- `smoke/`: preflight results, separate from production outcomes.

Execution metadata records the eight actors, independent seed scheme, central
merge order and cache/transfer settings. Trajectory rows additionally record
the **last player phase's** collection time and cache counts; these are not
whole-iteration timings. The existing RSS column measures the central learner
only, explicitly recorded in the manifest; it must not be interpreted as
total Ray/actor memory. Output loaders reject mismatched execution metadata.
There are no replay/optimizer dumps or resumable training states.

As in Experiment 4, checkpoints and stopping occur after complete outer
iterations. The 24-hour active clock includes collection, fitting, archive
capture, and (conservatively) one-time lazy Ray/actor startup. Checkpoint
serialization, reload validation and upload are excluded. Startup is not
silently subtracted from the parallel arm's budget.

Use the same pushed code revision and dependencies for both Experiments 4 and
5 when possible. This revision also contains common encoding/replay efficiency
improvements; comparing a newly built parallel arm against an older sequential
revision would mix those changes with parallel execution. Existing runs are
not overwritten, restarted or silently re-labelled.

Primary efficiency measures are nodes per active hour and time to a common
node boundary (interpolated only within observed ranges). Because VM allocation
is identical, a measured throughput improvement also improves node throughput
per nominal VM-hour. Policy quality still needs checking at common nodes and
common active time using the unchanged sampled metrics; LBR is not exact
exploitability. Equal seed labels do not make serial/parallel trajectories
identical or remove training variability. No automatic Experiment 4 download
or new cross-experiment match league is introduced; the preserved output schemas
allow the later comparative analysis.

## Launch after committing and pushing

Use the same PROJECT_ID, REGION, BUCKET and SA_EMAIL as Experiment 4.
The runner must retain its existing child-job and service-account-use grants.

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr5-par8-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=36
bash gcp/run_exp5_sd_cfr_parallel_24h.sh run
```

The remote controller runs smoke, three training VMs, aggregation,
evaluation-cost profiling and standalone evaluation. The laptop may disconnect
after successful submission. Completion takes more than 24 elapsed hours due
to setup, checkpoint I/O and subsequent evaluation. The evaluation cost gate
retains training results and stops for review if the projected cost exceeds
EVAL_MAX_HOURS; this does not change the training budget.

```bash
bash gcp/run_exp5_sd_cfr_parallel_24h.sh status
bash gcp/run_exp5_sd_cfr_parallel_24h.sh dry-run
# Optional, requires installed project dependencies and local Ray processes:
bash gcp/run_exp5_sd_cfr_parallel_24h.sh smoke-local
# With the original RUN_ID: retry evaluation without retraining.
bash gcp/run_exp5_sd_cfr_parallel_24h.sh evaluate-only
```

No job is submitted by tests or dry-run. A stale REPO_REF without Experiment 5
and its parallel solver is rejected before cloud submission.
