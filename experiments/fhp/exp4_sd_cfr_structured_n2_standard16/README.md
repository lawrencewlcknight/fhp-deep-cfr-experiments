# Experiment 4: structured SD-CFR on n2-standard-16

This is a **hardware-only counterpart to Experiment 3**. It trains the identical
structured-input SD-CFR learner on a larger VM. It does not add parallel
traversal, increase Torch threads, enlarge the networks, or change optimisation.

## Configuration

| Setting | Experiment 3 | Experiment 4 |
| --- | --- | --- |
| Training VM, per seed | n2-standard-8 | **n2-standard-16** |
| vCPUs / memory | 8 / 32 GiB | **16 / 64 GiB** |
| Independent training seeds | 0, 1, 2 | Unchanged |
| Active training budget, per seed | 24 hours | Unchanged |
| Playable policy checkpoints | 6, 12, 18, 24 hours | Unchanged |
| Torch intra-op / inter-op threads | 1 / 1 | Unchanged |
| Traversal implementation | Sequential | Unchanged |
| Disk / provisioning | 200 GiB pd-balanced / STANDARD | Unchanged |

Training settings are imported from Experiment 3: its exact 183-feature
player-observable suit-canonical encoder, 8 x 32 residual/layer-normalised
advantage networks, 320 traversals/player/iteration, 200 advantage updates,
minibatches of 2,048, Adam learning rate 0.004, continuous warm start, target
standardisation, five-million-row replay per player, and uniform historical
strategy weighting. The same structured solver and lossless two-bit replay
are used. There is no average-policy network or new privileged information.

Seeds run concurrently on **three separate n2-standard-16 VMs**, not three tasks
packed onto one VM. Peak training allocation is **48 N2 vCPUs**, plus the small
controller. Batch requests 16,000 milli-CPUs and 60,000 MiB per worker, leaving
headroom within the VM's 64 GiB. The cloud smoke uses the same larger VM.
The controller remains e2-small; aggregation, evaluation-cost profiling and
evaluation remain n2-standard-8 with the same eight evaluation workers.
Auxiliary stages therefore do not confound training-throughput comparisons.

The larger VM is not expected to deliver twice the throughput automatically:
the preserved learner is sequential and uses one Torch thread. More RAM and
vCPUs do not by themselves parallelise this computation. The experiment measures
whether this VM allocation improves useful training throughput; unchanged
throughput is a meaningful result. A multi-worker algorithm would require a
separate experiment.

## Output and comparison contract

The baseline output structure is preserved, under a distinct run ID:

- `workers/`: run/hardware/encoder manifests, training trajectory, losses,
  diagnostics, four playable policy prefixes and immutable float32 historical
  network chunks stored once.
- `analysis/`: checkpoint index, node/time summary and throughput chart.
- `evaluation/`: five rule-agent evaluations, LBR diagnostics, temporal
  head-to-head results, and quality charts indexed by time and nodes.
- `smoke/`: configuration, encoder, solver-equivalence, capacity and evaluator
  checks, clearly separate from production results.

Experiment and report identifiers are distinct even though the algorithm ID
and seed task names are unchanged. Outputs must not share an Experiment 3
run directory. Loading Experiment 3 results through the Experiment 4 evaluator
is rejected, avoiding silent cohort mixing.

No full replay/optimizer states are retained; checkpoints support evaluation,
not resumed training. Active time excludes checkpoint serialization, reload
validation and upload, but includes historical network capture, just as in
Experiment 3. Checkpoints and stopping occur at completed outer iterations.

The standalone evaluation budgets, deal seeds, LBR settings and protocols are
unchanged. LBR is not exact exploitability. No outputs from other experiments
are downloaded; cross-experiment comparisons remain retrospective work. Compare
nodes per active hour, per-node learning quality, and measured compute cost.
Run Experiments 3 and 4 from the same code revision/dependency environment when
possible, and retain their manifests; physical host variation remains a source
of timing noise. Identical settings do not guarantee bitwise-identical long
trajectories on different CPUs or identical node counts at a time limit.

## Launch after committing and pushing

Use the Deep CFR bucket and runner, not UCV inputs. Set PROJECT_ID, REGION,
BUCKET and SA_EMAIL as for Experiment 3. The controller service account needs
`roles/batch.jobsEditor` on the project and `roles/iam.serviceAccountUser` on
itself, in addition to its existing worker/logging/storage permissions. These
are the same controller permissions required by Experiments 2 and 3; changing
VM size does not fix a missing IAM grant. The existing launcher checks that
the service account exists, not that all of these grants are present.

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr4-vm16-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=36
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh run
```

Pipeline: smoke, three parallel training VMs, aggregation, evaluation-cost
profile, standalone evaluation. The cost gate stops for review if estimated
evaluation exceeds EVAL_MAX_HOURS, retaining training results. Completion takes
more than 24 elapsed hours because setup, checkpoint I/O and evaluation are
additional. The laptop may disconnect after controller submission.

```bash
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh status
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh dry-run
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh smoke-local
# With the original RUN_ID: resume evaluation only, without retraining.
bash gcp/run_exp4_sd_cfr_structured_n2_standard16.sh evaluate-only
```

Local smoke requires the project's Python dependencies and does not test cloud
IAM or n2-standard-16 hardware. No cloud jobs are launched by the unit tests.
