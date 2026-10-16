"""Opt-in robust artifact template aggregators; never changes historical mean."""
from __future__ import annotations

import numpy as np


def aggregate_template(rows: np.ndarray, method: str = "mean", *,
                       trim_fraction: float = 0.0, huber_delta: float = 1.5,
                       huber_iterations: int = 12) -> np.ndarray:
    """Aggregate artifact segments sample-wise, using a robust estimator."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("rows must be a finite, nonempty 2-D array")
    if method not in {"mean", "trimmed", "median", "huber"}:
        raise ValueError(f"Invalid template method: {method!r}")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if huber_delta <= 0 or not np.isfinite(huber_delta):
        raise ValueError("huber_delta must be finite and positive")
    if method in {"mean", "trimmed"}:
        cut = int(np.floor(trim_fraction * len(x)))
        if cut <= 0 or 2 * cut >= len(x):
            return x.mean(axis=0)
        return np.sort(x, axis=0)[cut:len(x) - cut].mean(axis=0)
    if method == "median":
        return np.median(x, axis=0)

    estimate = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - estimate), axis=0)
    scale = np.maximum(scale, np.finfo(np.float64).eps * np.maximum(1, np.abs(estimate)))
    threshold = huber_delta * scale
    for _ in range(huber_iterations):
        deviation = np.abs(x - estimate)
        weights = np.minimum(1.0, threshold[None, :] /
                             np.maximum(deviation, np.finfo(float).tiny))
        updated = (weights * x).sum(axis=0) / weights.sum(axis=0)
        if np.allclose(updated, estimate, atol=0, rtol=1e-9):
            break
        estimate = updated
    return estimate
