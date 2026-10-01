# Testing

Run the repository checks from its root:

```bash
pytest
```

The tests pin the canonical FHP parameters and OpenSpiel game shape, exercise
the SD-CFR efficiency audit (Experiment 1) and the archived conventional Deep CFR
runner end to end without full-tree evaluation, and
reject stale variant names or committed output artefacts.

Experiment 2 tests additionally cover the three-seed/time-budget contract,
lossless disk archive parity and bounded storage, immutable prefix reloads,
checkpoint-clock exclusion, own-reach mixture equivalence (including a case
where a naive pointwise average is provably wrong), independent per-hand
strategy draws, retrospective deal streams, all nine cross-seed matchups,
seed-cluster inference, comparative reporting and generated Batch shell syntax.

```bash
python -m pytest -q tests/test_exp2_sd_cfr_24h.py
bash gcp/run_exp2_sd_cfr_24h.sh smoke-local
```

The cloud gate additionally reruns the six-iteration implementation audit on
seeds 0/1/2 (one repeat), touches production-capacity replay, reloads a 10,000-
iteration synthetic archive, and smoke-tests the real UCV comparator policies.
These checks cannot establish 24-hour speed, policy quality or real-archive
evaluation cost in advance; the experiment measures them.
