# Experiment 3: exact UCV Experiment 2 player inputs for SD-CFR

## Controlled change

Does the lossless, suit-canonical player-information representation from FHP
UCV-ESCHER Experiment 2 improve SD-CFR over raw-input SD-CFR Experiment 2?

This transfers only the **183-value player input**, not the UCV solver or its
branched network. The `fhp_lossless_suit_canonical_v1` feature order is:

- 104 exact private/public card indicators, with suits canonicalised jointly;
- 2 player indicators, 2 round indicators and 30 exact betting-history values;
- 26 normalised private/public rank counts and 8 normalised suit counts;
- 2 private-hand flags (pocket pair and suitedness);
- 9 exact five-card hand-category indicators after the flop.

Rank identities and private/public roles remain distinguishable. Only global
suit-name symmetry is merged; no hand buckets or best-response information
are used. No opponent private cards, critic full-state input, or feature
extensions from later UCV experiments are included.

Source: `fhp_escher/features.py` in `lawrencewlcknight/fhp-ucv-escher-experiments`,
commit `54a3269f62189b8ac7190e59c9a3ea70efb4969c`, SHA256
`9c799a9b24465dce30de5a9d50084893b086d85445f7670425ba2147266cb132`.
84 golden observations generated with that encoder cover every made-hand
category, both players, both rounds, and varied betting sequences. Tests
require exact float32 agreement, all 24 suit permutations, and independence
from the opponent's actual cards. The source repository is not needed at runtime.

## Unchanged training contract

All training settings are imported from SD-CFR Experiment 2:

- **Three seeds: 0, 1, 2**, concurrently on separate on-demand **n2-standard-8**
  VMs (24 N2 vCPUs, plus a small controller).
- **24 active hours per seed**, stopping at a completed outer iteration.
- Playable policies at the first completed iteration crossing **6, 12, 18, 24 hours**.
- **8 x 32 residual/layer-normalised centred-advantage networks**. Only the
  input dimension changes from 190 to 183; hidden architecture and widths are
  unchanged. Parameter counts therefore differ slightly.
- 320 traversals/player/iteration; 200 advantage updates/player/iteration;
  minibatches 2,048; constant Adam learning rate 0.004; continuous warm start
  including optimizer state; the same target standardisation and regression weights.
- 5,000,000 uniformly sampled replay rows/player; uniform historical-strategy
  weighting; all iteration networks retained; no average-policy fitting.
- One Torch intra-op/inter-op thread. Same active clock, safety cap, checkpoint
  exclusions, machine/disk allocation and elapsed-time limits.

This representation is intended to change learning outcomes; it should not
reproduce raw-input policies bit for bit. Tests instead compare packed and
ordinary float32 replay **within the new representation**, requiring identical
losses, weights, node counts and RNG states.

## Storage and evaluation

The encoder includes fractions (e.g. one third). Replay stores two-bit integer
numerators with fixed per-feature denominators, recovering the original float32
values exactly. Unknown values fail rather than being rounded. Inputs occupy
**46 bytes/row**, versus 24 for raw binary inputs or 732 for dense float32
structured inputs. Both full replay buffers allocate approximately **620 MB**
including targets and iteration labels.

Training, sampled historical-policy play and exact own-reach mixture queries
all use the versioned encoder. Archives reject missing/incompatible metadata;
raw Experiment 2 policies still use the raw path. Immutable float32 historical
weight chunks are stored once and shared by checkpoint prefixes. No full replay
or optimizer states are uploaded. Checkpoints support play/evaluation, **not
resumption of learning**.

The standalone evaluation protocol is unchanged: five rule agents (10,000
duplicate pairs each/checkpoint), LBR (1,000 pairs/checkpoint, 4,096 preflop
rollouts), and every within-seed earlier/later checkpoint pair (50,000 pairs
each). Baseline evaluation seeds are retained; tables/charts report quality by
time and nodes. LBR is not exact exploitability. The same evaluation-cost gate
applies: estimates above EVAL_MAX_HOURS stop for review, preserving trained
policies. No other experiment's outputs are downloaded; cross-experiment
comparisons are deferred.

Assess both equal-time and node-indexed results: feature computation has a cost,
so better learning per node need not give the same improvement per hour. No
speed or policy-quality gain is assumed.

## Launch after committing and pushing

With PROJECT_ID, REGION, BUCKET and SA_EMAIL already configured:

```bash
git pull --ff-only
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="sdcfr3-24h-$(date -u '+%Y%m%d-%H%M%S')"
export EVAL_MAX_HOURS=36
bash gcp/run_exp3_sd_cfr_structured_24h.sh run
```

BUCKET remains the Deep CFR results bucket. No UCV bucket permissions/run IDs
are needed. Pipeline: cloud feature/parity/storage smoke, three parallel
training VMs, aggregation, cost profile, standalone evaluation. The laptop can
disconnect after controller submission. Experiment 2 is not retrained or modified.

```bash
bash gcp/run_exp3_sd_cfr_structured_24h.sh status
bash gcp/run_exp3_sd_cfr_structured_24h.sh dry-run
bash gcp/run_exp3_sd_cfr_structured_24h.sh smoke-local
# Same RUN_ID: resume evaluation only, never repeat training.
bash gcp/run_exp3_sd_cfr_structured_24h.sh evaluate-only
```

Outputs under `gs://BUCKET/RUN_ID/` retain the baseline structure: `workers/`
contains playable archives, manifests, losses, throughput and diagnostics;
`analysis/` contains training tables/charts; `evaluation/` contains resumable
task results and rule/LBR/temporal tables and charts; `smoke/` contains checks.
Manifests record the distinct experiment ID and encoder metadata.
