# SD-CFR port for FHP

## What was ported

This repository previously contained only conventional Deep CFR. The SD-CFR
archive and policy implementations have been copied from
`leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments/deep_cfr_poker/sd_cfr.py`
at source commit `1669e5af4cbc88c648626148fd9c395c2e5d4583`.
The original file SHA-256 is
`52b09ad06454b1b282b08558723b13d2672c89420eaddbb20af6f8a6c0e7ae4f`.

The default recipe is **Leduc Experiment 28's selected uniformly weighted
SD-CFR**, used standalone in Experiment 29. It is not Experiment 30's
paper-aligned higher-traversal/fresh-optimiser recipe. The two 36-hour schedules
were close at their final endpoint; the selected schedule was stronger at the
common node boundary. Here “selected” identifies the existing thesis comparator,
not a claim that it is already optimal for FHP.

| Training setting | Value |
|---|---|
| Traversals per player/iteration | 320 |
| Advantage architecture | Residual, LayerNorm, centred advantage; 8 x 32 |
| Advantage updates per player/iteration | 200 |
| Advantage minibatch | 2,048 |
| Learning rate | Constant 0.004 |
| Network and Adam state | Continuous warm start |
| Advantage replay | Uniform reservoir; 5 million rows per player |
| Advantage targets | Batch standardisation, epsilon 1e-6 |
| Advantage loss weighting | Iteration-weighted, unchanged from the source |
| Output-policy mixture | Uniform over retained iterations |
| Average-policy fitting / strategy replay | Disabled |

Uniform **output mixture** weighting does not remove the iteration weighting
from the advantage regression objective. Input dimensionality now follows
the canonical FHP game (190 features, three actions); the network structure
otherwise remains unchanged. FHP's existing float32/int32 compact replay and
inference-mode traversal code are preserved. No card abstraction or new features
are introduced by this port.

The shared solver receives only additive support for disabling policy fitting,
skipping strategy replay, per-player archive callbacks and a time limit. Existing
Deep CFR defaults and the archived conventional experiment's algorithm settings
are unchanged. The active Experiment 1 now benchmarks SD-CFR execution efficiency.
An empty
one-slot strategy buffer and unused small policy module remain for constructor
and RNG compatibility; no strategy records, policy updates or Adam moments for
that module are generated. The standalone result returns `policy_network=None`
and rejects exporting the unused module as if it were a trained policy.

## Training and playable archives

`SingleDeepCFRSolver` captures an immutable CPU copy immediately after each
player's advantage update, matching the source's alternating-update schedule.
The archive has every historical network for both players. It is **not** a
single final regret network and is not reduced by checkpoint thinning.

Minimal local smoke example (not a production experiment):

```python
import torch
from deep_cfr_poker import SingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed

torch.set_num_threads(1)
set_seed(1234)
solver = SingleDeepCFRSolver(
    num_iterations=2, num_traversals=2,
    advantage_network_layers=(8, 8), policy_network_layers=(8, 8),
    memory_capacity=256, batch_size_advantage=2,
    advantage_network_train_steps=1, evaluation_interval=1,
)
result = solver.solve()
solver.save_archive("/tmp/fhp_sd_cfr_smoke.pt")
policy = solver.make_policy(seed=42)
# Before EACH independent hand:
policy.resample_episode()
# At every decision in that hand:
# probabilities = policy.action_probabilities(state)
```

`solve(max_training_seconds=...)` optionally stops after a complete two-player
iteration, including archive-capture/callback costs in the elapsed budget.
The iteration cap still applies. No production duration, seed allocation,
cloud launcher or checkpoint cadence has been introduced by this port.

`save_archive()` stores game identity, architecture, weighting metadata and
historical network weights, **not replay buffers or optimisers**. It is a
playable artefact, not a resumable training checkpoint. Use `SDCFRArchive.load`
and `SampledSDCFRPolicy(..., weighting="uniform")` to reload it. The generic
low-level policy retains the source's default `linear` weighting for API
compatibility; `solver.make_policy()` correctly supplies the selected uniform
weighting. Optional archive prefixes support evaluation at earlier iterations.

## Evaluation constraints

Each player samples one historical network using the chosen iteration weights
at the start of the hand, then keeps it throughout the hand. Drawing a new
network at each decision or averaging action probabilities without own-reach
weights implements a different policy. Both-seat duplicate play needs a fresh
episode sample for each seat-swapped hand and a controlled separate model RNG.

Do **not** pass the low-level `SampledSDCFRPolicy` to a generic average-policy
evaluator without an episode-reset adapter. Experiment 2 now provides
`DiskSampledPolicy.begin_episode` and an optional hook in the pinned duplicate
evaluator. LBR queries a separate `DiskBehaviouralPolicy`: the exact own-reach
mixture evaluated at the requested history, not the secretly sampled model.
Its performance on large real archives is guarded by an explicit cloud cost
profile before full evaluation. The conventional checkpoint loader still
rejects SD-CFR archives instead of interpreting them as average networks.

Exact behavioural reconstruction helpers are retained for small-game regression
tests, but explicitly fail for `universal_poker` before enumerating any game tree.
No exact exploitability calculation is required to train or play on FHP.

The low-level reference API keeps networks in a CPU archive, as in the source.
Experiment 2 replaces only that storage layer with lossless immutable disk
chunks and checkpoint-prefix manifests. It bounds in-memory history storage,
loads two episode-selected networks for sampled play, and saves historical
weights only once. Disk storage still grows with iterations; unlike a distilled
average network, this is not a constant-size policy. No replay/optimiser dumps
are retained and these playable checkpoints cannot resume training.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
```

Coverage includes small-game exact-versus-historical-mixture equivalence for
uniform and linear weights, phase-correct immutable archive capture, inference
RNG isolation, checkpoint reloads, compact/Python replay parity, full FHP hands,
opponent-card privacy, no average-policy training/replay, complete-iteration
time stops and full-tree guards. Existing conventional Deep CFR tests remain.

A separate fixed-seed parity check ran the original source and the FHP port on
the same FHP game for three iterations, using the selected 8 x 32 advantage
architecture with reduced smoke budgets. Both Python and compact FHP replay
produced bit-identical archived network tensors and identical losses and node
counts (109, 274, 459). This verifies the port on that test, not long-run FHP
performance or bitwise identity across different hardware/software runtimes.
