# Testing

Run the repository checks from its root:

```bash
pytest
```

The tests pin the canonical FHP parameters and OpenSpiel game shape, exercise
a tiny Deep CFR training pass without full-tree evaluation, and reject stale
variant names or committed experiment artefacts.
