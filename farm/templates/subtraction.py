"""Adaptive artifact-template construction and scaling (FARM step v).

This implementation deliberately stays close to the published/legacy FARM
behaviour: for each acquisition group, choose the most correlated nearby
occurrences, average them, and fit one scalar amplitude.  The important timing
fix retained here is that the *last* acquisition group is correlated over its
full segment.  The inter-volume dead time starts after that segment and must
not be subtracted from the correlation window.
"""

from __future__ import annotations

import numpy as np

from farm.utils.signal import corr_with_matrix


def _trimmed_mean_rows(x: np.ndarray, trim_fraction: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a non-empty 2-D array")
    n_trim = int(np.floor(float(trim_fraction) * x.shape[0]))
    if n_trim <= 0 or 2 * n_trim >= x.shape[0]:
        return x.mean(axis=0)
    ordered = np.sort(x, axis=0)
    return ordered[n_trim:x.shape[0] - n_trim].mean(axis=0)


def build_artifact_templates(
    aligned_segments: np.ndarray,
    valid_idx: np.ndarray,
    slice_info: dict,
    n_candidates: int,
    dtime_samp_up: int,
    seg_len: int,
    *,
    trim_fraction: float = 0.0,
    min_correlation: float = -1.0,
    scale_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build one artifact template per valid acquisition-group occurrence.

    Defaults reproduce the legacy FARM selection as closely as possible while
    keeping the corrected full correlation window for the last group.
    Optional robust trimming/correlation gating can still be enabled from the
    configuration for difficult datasets.
    """
    del dtime_samp_up  # dead time lies after the final group, not inside it

    aligned_segments = np.asarray(aligned_segments, dtype=np.float32)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if aligned_segments.ndim != 2:
        raise ValueError("aligned_segments must be a 2-D array")
    if len(aligned_segments) != len(valid_idx):
        raise ValueError("aligned_segments and valid_idx length mismatch")
    if not (0.0 <= float(trim_fraction) < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")

    row_of_slice = {int(s): row for row, s in enumerate(valid_idx)}
    artifact_segments = np.zeros_like(aligned_segments)
    window = slice(0, int(seg_len))

    for row, slice_idx in enumerate(valid_idx):
        candidates = np.asarray(slice_info["candidate_idx"][slice_idx], dtype=np.int64)
        candidates = candidates[(candidates >= 0) & (candidates != slice_idx)]
        candidate_rows = np.asarray(
            [row_of_slice[int(c)] for c in candidates if int(c) in row_of_slice],
            dtype=np.int64,
        )
        if len(candidate_rows) < 2:
            continue

        target = aligned_segments[row].astype(np.float64, copy=False)
        candidate_data = aligned_segments[candidate_rows].astype(np.float64, copy=False)
        correlations = corr_with_matrix(target[window], candidate_data[:, window])

        n_keep = min(int(n_candidates), len(candidate_rows))
        order = np.argsort(correlations)[::-1]
        top = order[:n_keep]
        if float(min_correlation) > -1.0:
            top = top[correlations[top] >= float(min_correlation)]
        if len(top) < 2:
            continue

        selected_rows = candidate_rows[top]
        template = _trimmed_mean_rows(
            aligned_segments[selected_rows], trim_fraction=float(trim_fraction)
        )

        denominator = float(np.dot(template[window], template[window]))
        if denominator < 1e-20:
            continue
        scaling = float(np.dot(target[window], template[window]) / denominator)
        if scale_bounds is not None:
            scaling = float(np.clip(scaling, scale_bounds[0], scale_bounds[1]))

        artifact_segments[row] = (scaling * template).astype(np.float32)

    return artifact_segments
