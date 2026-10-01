"""Uncertainty summaries for independent duplicate-deal pairs."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats


def sample_summary(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty, finite one-dimensional sample")
    n = int(array.size)
    mean = float(np.mean(array))
    if n == 1:
        return {"n": n, "mean": mean, "std": math.nan, "se": math.nan,
                "ci95_low": math.nan, "ci95_high": math.nan}
    std = float(np.std(array, ddof=1))
    se = std / math.sqrt(n)
    margin = float(scipy_stats.t.ppf(0.975, n - 1)) * se
    return {"n": n, "mean": mean, "std": std, "se": se,
            "ci95_low": mean - margin, "ci95_high": mean + margin}

