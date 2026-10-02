# FHP Deep CFR Experiments

## Shared policy evaluation

Conventional FHP snapshots use the shared `fhp-evaluation-suite` API. A pinned
snapshot of that evaluator is bundled for reproducible cloud runs; its
provenance and minimal SD-CFR extensions are in `fhp_evaluation/VENDORED_FROM.md`.
The conventional snapshot adapter is
`deep_cfr_poker.evaluation_adapter`. Install it from this directory with
`python -m pip install -e ../../fhp-evaluation-suite`; then run
`fhp-evaluate benchmark SNAPSHOT --deals 10000 --seed 2026` or
`fhp-evaluate lbr SNAPSHOT --deals 1000 --seed 2026`.

The benchmark uses both-seat duplicate deals and corrected LooseAggressive
bands `(-300,-100)`. LBR is reported as a lower bound, not exact exploitability.

## SD-CFR implementation

The selected **uniform-weighted SD-CFR** implementation from Leduc Experiments
28/29 is now available as `deep_cfr_poker.SingleDeepCFRSolver`. It retains the
historical advantage networks instead of fitting an average-policy network.
The FHP game and existing compact replay/traversal optimisations are retained.

See [SD-CFR usage, provenance and evaluation restrictions](docs/SD_CFR.md).
The former conventional Deep CFR Experiment 1 is archived; the active
Experiment 1 is now the SD-CFR efficiency audit below, not a long-horizon
training run. SD-CFR archives are not conventional policy snapshots
and must not be passed to the average-policy evaluation commands above.

## Experiment 1: SD-CFR efficiency audit

A short three-seed, three-repeat fixed-work benchmark compares the reference
SD-CFR implementation with live scripted inference and with scripted inference
plus lossless binary replay compression. It checks networks, replay, optimiser,
RNG states and playable policy outputs before interpreting speed measurements.
Production solver defaults are unchanged. See the
[experiment specification and local/GCP commands](experiments/fhp/exp1_sd_cfr_efficiency/README.md).

## Experiment 2: optimised SD-CFR, 24 hours

Three seeds (`0, 1, 2`) train concurrently on separate `n2-standard-8` VMs for
24 active hours each, retaining playable policies at 6, 12, 18 and 24 hours.
The selected uniform algorithm is unchanged: scripted live inference and
losslessly packed replay improve implementation efficiency. Historical
networks are stored once in bounded-memory chunks; no full replay dumps are
retained. The output is the historical mixture, never the final network alone.

The cloud workflow runs equivalence, capacity and evaluator smoke checks,
then training, aggregation, an evaluation-cost profile, and standalone rule-agent,
LBR and temporal head-to-head evaluation. No UCV checkpoints, previous evaluation
outputs or cross-bucket access are required; cross-algorithm comparisons are deferred.
LBR uses the exact own-reach mixture
at queried histories, without enumerating the game tree. A failed cost gate
preserves all training outputs and stops before full evaluation.

See [Experiment 2 configuration and launch instructions](experiments/fhp/exp2_sd_cfr_24h/README.md).

## Experiment 3: suit-canonical structured inputs, 24 hours

Experiment 3 retains Experiment 2's three seeds, 24-hour budgets, hardware,
8 x 32 residual networks and standalone evaluation. Only the input representation
changes to the exact 183-value **player-observable** encoder from FHP UCV-ESCHER
Experiment 2. This includes lossless suit canonicalisation and derived poker
features, but not the UCV network architecture, critic input, or later features.
Compact two-bit replay preserves every float32 input exactly; historical-policy
evaluation uses the versioned encoder too. No other experiment outputs are required.

See [Experiment 3 specification and launch instructions](experiments/fhp/exp3_sd_cfr_structured_24h/README.md).

Fresh experiment repository for applying Deep Counterfactual Regret
Minimisation (Deep CFR) to flop hold'em poker (FHP).

## Canonical game contract

The repository uses OpenSpiel's `universal_poker` implementation with this
exact two-player, two-round limit configuration:

| Parameter | Value |
| --- | --- |
| `blind` | `50 100` |
| `raiseSize` | `100 100` |
| `firstPlayer` | `1 2` |
| `maxRaises` | `3 3` |
| `numRounds` | `2` |
| `numSuits` | `4` |
| `numRanks` | `13` |
| `numHoleCards` | `2` |
| `numBoardCards` | `0 3` |
| `numPlayers` | `2` |

This contract is pinned in `deep_cfr_poker/game.py` and matches the existing
FHP repository in this workspace, including its VR-Deep provenance commit.

## Archived conventional Deep CFR experiment

The superseded experiment is retained under
`archive_exp1_deep_cfr_best_config_transfer` for historical reference only.
It transfers the previously selected Deep CFR configuration to
FHP over five fixed seeds. It uses 1,050 iterations, 320 traversals per player,
the `2x32` policy network, the residual LayerNorm centred-advantage `8x32`
network, and the approved optimiser, replay, and policy-fitting settings.

The implementation, smoke command, full command, output contract, and Google
Cloud Batch command are documented in
`experiments/fhp/archive_exp1_deep_cfr_best_config_transfer/README.md`.
Its historical cloud workflow is retained in
`docs/archive_GCP_DEEP_CFR_EXPERIMENT1.md`; it is not the active Experiment 1.

For complete project setup, IAM, smoke testing, monitoring, logs, output
retrieval, resource sizing, and cleanup instructions, see
`docs/GCP_BATCH_EXPERIMENTS.md`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest
```

Exact full-tree exploitability and policy enumeration are disabled for normal
FHP training because the game tree is too large. Experiment checkpoints should
be saved for sampled head-to-head evaluation.
