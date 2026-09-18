"""Adaptive template construction and scaling (FARM step v)."""

import numpy as np

from farm.utils.signal import corr_with_matrix


def build_artifact_templates(
    aligned_segments: np.ndarray,
    valid_idx: np.ndarray,
    slice_info: dict,
    n_candidates: int,
    dtime_samp_up: int,
    seg_len: int,
) -> np.ndarray:
    """Build one artifact template per valid slice-group segment.

    Candidate templates are expected to originate from the same slice-group
    across nearby volumes, as produced by ``build_slice_info``.
    """
    aligned_segments = np.asarray(aligned_segments, dtype=np.float32)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    if aligned_segments.ndim != 2:
        raise ValueError("aligned_segments must be a 2-D array")

    if len(aligned_segments) != len(valid_idx):
        raise ValueError("aligned_segments and valid_idx length mismatch")

    row_of_slice = {int(slice_idx): row for row, slice_idx in enumerate(valid_idx)}
    artifact_segments = np.zeros_like(aligned_segments)

    for row, slice_idx in enumerate(valid_idx):
        candidates = np.asarray(
            slice_info["candidate_idx"][slice_idx],
            dtype=np.int64,
        )

        candidates = candidates[candidates >= 0]
        candidates = candidates[candidates != slice_idx]

        candidate_rows = np.asarray(
            [
                row_of_slice[int(candidate)]
                for candidate in candidates
                if int(candidate) in row_of_slice
            ],
            dtype=np.int64,
        )

        if len(candidate_rows) < 2:
            continue

        # Do not include the dead-time transition in the correlation window
        # for last groups. Usually the new zero-fill means this matters less,
        # but retaining the guard is harmless.
        if bool(slice_info["is_last"][slice_idx]):
            window = slice(0, max(seg_len - max(0, dtime_samp_up), 8))
        else:
            window = slice(0, seg_len)

        target = aligned_segments[row].astype(np.float64)
        candidate_data = aligned_segments[candidate_rows].astype(np.float64)

        correlations = corr_with_matrix(
            target[window],
            candidate_data[:, window],
        )

        order = np.argsort(correlations)[::-1]
        selected_rows = candidate_rows[
            order[:min(int(n_candidates), len(candidate_rows))]
        ]

        if len(selected_rows) < 2:
            continue

        template = aligned_segments[selected_rows].mean(axis=0).astype(np.float64)

        denominator = float(np.dot(template[window], template[window]))
        if denominator < 1e-20:
            continue

        scaling = float(
            np.dot(target[window], template[window]) / denominator
        )

        artifact_segments[row] = (scaling * template).astype(np.float32)

    return artifact_segments