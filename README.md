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

## Repository state

This is a clean experiment series. There are no inherited training outputs,
experiment results, checkpoints, charts, or experiment-number history.

Experiment 1 is intentionally reserved until its proposed configuration has
been reviewed and approved. No training run has been started.

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
