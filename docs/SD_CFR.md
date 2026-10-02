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

The opt-in parallel backend and its separate real-process tests are documented
below. The ordinary suite does not start a Ray cluster.

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

## Parallel traversal solver

`ParallelSingleDeepCFRSolver` and `ParallelStructuredSingleDeepCFRSolver`
adapt the existing parallel Deep CFR central-learner/Ray-worker design to
SD-CFR. This is a reusable implementation, not a new numbered experiment.
Experiments 2, 3 and 4 retain their sequential defaults and cloud launchers.

What remains unchanged:

- Total traversals **per player**, not per worker: 320 split over eight
  workers means 40 each, not 2,560. Uneven divisions retain the exact total.
- The alternating update order: collect player 0, fit and archive player 0,
  broadcast updated weights, collect player 1, fit and archive player 1.
- Central uniform reservoir sampling, iteration-weighted advantage regression,
  target processing, float32 targets, warm-start networks and persistent Adam.
- All historical advantage networks are retained. There is no average-policy
  fitting, strategy replay, archive thinning or information leakage.
- Raw 190-value inputs, or the exact 183-value observable structured encoder;
  their replay encodings are lossless, not reduced-precision approximations.

Unlike the older Deep CFR workers, these workers do **not** keep local
reservoirs before merging. They return every observation in bounded, typed
chunks. Only the central learner applies Algorithm R. This avoids a second
sampling stage and avoids allocating a production-size replay on each worker.
Features are batch-packed before transfer, and stay packed during central
insertion. Each phase broadcasts one immutable weight snapshot to all actors.
Results are merged in worker-index/traversal order, not completion order.

Each actor uses one Torch thread. Neural optimisation remains on the central
learner; this is not distributed gradient training. Start with one Torch thread
on the learner for reproducible comparisons. More workers can reduce collection
time, but cannot accelerate the serial fitting and archive stages. Ray startup,
communication, feature encoding and central replay insertion also cost time.
Measure end-to-end speed on the intended VM; no speedup factor is assumed.

Minimal structured-input smoke example (change class for raw inputs):

```python
import torch
from deep_cfr_poker import ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed

torch.set_num_threads(1)
set_seed(1234)  # learner initialization, replay sampling and optimisation
with ParallelStructuredSingleDeepCFRSolver(
    parallel_num_workers=2,
    parallel_run_seed=1234,  # explicit traversal streams
    num_iterations=3, num_traversals=8,
    memory_capacity=256, batch_size_advantage=8,
    advantage_network_train_steps=2,
) as solver:
    result = solver.solve()
    solver.save_archive("/tmp/fhp_parallel_sd_cfr_smoke.pt")
    print(solver.last_parallel_collection)
```

For a long run, retain the production recipe and attach `DiskSDCFRArchive`
as in Experiment 2 before `solve()`. It uses the same per-player capture
callback and supports existing playable checkpoint/evaluation adapters. Its
`checkpoint()` method, not `save_archive()`, writes disk-prefix manifests.
Neither representation is a resumable optimiser/replay checkpoint.

The local Ray runtime reserves one CPU per traversal actor. With an external
runtime, those resources must already be available; training waits at each
player boundary. `parallel_ray_address` can connect to an existing runtime,
but deploying code/dependencies to remote machines is the caller's job.
The context manager kills only this solver's actors and shuts down Ray only
when the solver started its own local runtime. Always use it (or `close()`).

Safety limits: chunks default to 4,096 rows, the per-worker/per-phase maximum
is 1,000,000 rows, and actor readiness/collection waits time out after 300
seconds. These are configurable via `parallel_chunk_rows`,
`parallel_max_rows_per_worker` and `parallel_timeout_seconds`. A limit or actor
failure raises an error and closes the solver; it never silently drops samples
or retries a partially completed iteration. The timeout does not bound Ray's
own initial `ray.init()` call. Phase metadata, shape, encoding, iteration and
finite targets are checked before central replay mutation. The default local
Ray object store is 512 MiB and can be changed with
`parallel_ray_object_store_memory`.

### Reproducibility and validation

Traversal seeds derive from the run seed, iteration, player and worker index.
For fixed worker count they do not depend on actor scheduling. Changing the
worker count or switching from the old sequential stream changes sampled
trajectories; bit-identical old-versus-parallel training is **not** promised.
The algorithm's sampling law and alternating learning rules remain unchanged.

`parallel_backend="serial"` runs the **same worker partitions and streams** in
one process as a correctness reference. It is not the original sequential
solver. The real-process test compares serial-reference and Ray outputs:
networks, replay, optimiser states, all archived weights, losses, node counts
and central RNG states. Both raw and structured variants were bit-identical
on the local smoke tests. Worker payloads are also tested independently against
the original traversal routine with matching phase seeds, and central packed
insertion against scalar reservoir insertion, including reservoir overflow.

```bash
# Ordinary suite, including non-Ray parallel correctness tests
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q

# Explicit real multi-process integration check; requires local sockets/processes
RUN_RAY_SD_CFR_TESTS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_sd_cfr_parallel.py -k real_ray
```

The integration test runs on an otherwise uninitialised Ray runtime and closes
it after each solver. These small tests do not establish long-horizon policy
quality, multi-host deployment or production VM speedup.

### Output-preserving efficiency review

- **Direct packed allocation:** the learner creates packed replay arrays
  immediately, rather than allocating and discarding dense float32 feature
  arrays first. At five million rows per player, those avoided feature arrays
  total 7.60 GB for raw inputs or 7.32 GB for structured inputs. NumPy can
  reserve these lazily: these are allocation sizes, not measured reductions
  in resident RAM. Steady-state packed replay size and values are unchanged.
- **No intermediate float32 code arrays:** packed worker batches stay uint8
  during validation and insertion. Algorithm R and its RNG order are unchanged.
- **Shared worker scratch space:** non-overlapping player phases reuse one
  staging workspace, halving its size (approximately 3 MB saved per worker at
  the default chunk size). Returned chunks own their data and are not overwritten.
- **Contiguous weight transport:** each phase sends two immutable float32
  arrays with validated layouts instead of many individual Torch tensors.
  Worker network storage receives exact copies; scripted inference still sees
  live weights. No parameter arithmetic or precision change occurs.
- **Bounded inference reuse:** exact repeated player inputs and legal-action
  masks reuse their regret-matched probabilities while networks are frozen.
  Full input bytes are compared, not approximate hashes or buckets. Caches
  clear at every player phase, including on errors. No sampled actions,
  chance outcomes, continuation returns or training targets are cached.
  `parallel_inference_cache_entries` defaults to 4,096 for structured inputs
  and zero for raw inputs, whose observed reuse was too low to justify the
  lookup overhead. Set it to zero for a cache-free comparison or if larger
  games have little repetition. Structured input keys occupy about 3 MB at
  4,096 entries, plus dictionary/result overhead, per worker.
- **Equivalent structured features and codec:** stable vectorised suit ordering
  replaces Python tuple construction; an 8 KiB rank-presence truth table
  replaces repeated straight tests; rank/suit counts are reused. Two-bit
  packing/unpacking uses integer lanes without intermediate per-bit arrays.
  Feature layout, representation identity and checkpoint compatibility stay
  unchanged. This is not an additional abstraction or hand-strength bucket.

Allocation and feature/codec improvements also benefit sequential optimised
SD-CFR. No numbered experiment's network, sampling, seeds, budget or parallelism
default changes.

Verification covers all 8,192 rank-presence patterns, suit-order ties, existing
encoder golden fixtures, byte-for-byte original/new codecs, shared-buffer
lifetimes, cache-off/on training, and actual multi-process Ray execution.
A saved pre-review implementation also matched bit-for-bit on six fixed-work
checks: three seeds per input representation, four iterations, 160 traversals
per player, two worker streams, production 8 x 32 networks, replay capacity
5,000, 20 updates and minibatches of 128. Checks compared replay, weights,
optimiser state, archives, losses, node counts and RNG state. Fitting budgets
were reduced; these are not long-run policy-quality experiments.

Short local serial-reference timings suggest modest savings, particularly for
structured inputs, but vary between repetitions. They exclude Ray startup
and do not establish an eight-worker VM speedup. Output identity is assessed
at **equal work**: faster fixed-time runs can complete more iterations and
finish with different policies. Existing traversal-time diagnostics and the
cache hit/miss fields in `last_parallel_collection` support VM profiling.

### Candidates requiring separate validation

1. **Batched traversal/GPU inference:** a promising next engineering candidate
   if profiling shows inference dominates. Process several traversal frontiers
   through a matrix batch, with per-trajectory RNG streams and explicit sample
   ordering. Batched kernels can change floating-point rounding and subsequent
   sampled actions. Validate fixed-work numerical behaviour first, then speed
   and policy quality on the target hardware. Not enabled here.
2. **Asynchronous collection with stale policies:** removing barriers can
   improve utilisation but changes the alternating update schedule. This needs
   an algorithmic experiment, not a silent implementation change.
3. **Mixed precision, smaller replay or historical-network thinning:** these
   change precision, training data or the deployed SD-CFR mixture. They need
   explicit approval and separate evaluation. Production runs should continue
   using the existing lossless disk archive.

The three-action loops deliberately retain ordered scalar floating-point sums:
there is little work to amortise and dot-product replacements can change
rounding. Neural minibatch training already uses matrix/tensor operations.
