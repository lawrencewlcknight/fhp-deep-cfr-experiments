# FHP Deep CFR Experiments

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

## Experiment 1

Experiment 1 transfers the approved best validated Deep CFR configuration to
FHP over five fixed seeds. It uses 1,050 iterations, 320 traversals per player,
the `2x32` policy network, the residual LayerNorm centred-advantage `8x32`
network, and the approved optimiser, replay, and policy-fitting settings.

The implementation, smoke command, full command, output contract, and Google
Cloud Batch command are documented in
`experiments/fhp/exp1_deep_cfr_best_config_transfer/README.md`. The full
training run has not been started.

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
