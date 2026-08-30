# Testing

Run the repository checks from its root:

```bash
pytest
```

The tests pin the canonical FHP parameters and OpenSpiel game shape, exercise
both the solver and Experiment 1 end to end without full-tree evaluation, and
reject stale variant names or committed output artefacts.
