# Testing

Run the repository checks from its root:

```bash
pytest
```

The tests pin the canonical FHP parameters and OpenSpiel game shape, exercise
the SD-CFR efficiency audit (Experiment 1) and the archived conventional Deep CFR
runner end to end without full-tree evaluation, and
reject stale variant names or committed output artefacts.
