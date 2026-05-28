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
    """Build one artifact template per valid slice segment.

    For each segment the *n_candidates* most-correlated neighbours
    (from the candidate window) are averaged, then least-squares
    scaled to the target segment.

    Parameters
    ----------
    aligned_segments : ndarray, shape ``(n_valid, seg_len)``.
    valid_idx : int64 array mapping rows to global slice indices.
    slice_info : dict from :func:`~farm.utils.slices.build_slice_info`.
    n_candidates : int — number of best candidates kept.
    dtime_samp_up : int — dead-time in upsampled samples.
    seg_len : int — segment length in upsampled samples.

    Returns
    -------
    artifact_segments : ndarray, same shape as *aligned_segments*.
    """
    row_of_slice = {int(idx): row for row, idx in enumerate(valid_idx)}
    artifact_segments = np.zeros_like(aligned_segments)

    for row, slice_idx in enumerate(valid_idx):
        cand = slice_info["candidate_idx"][slice_idx]
        cand = cand[cand >= 0]
        cand_rows = np.array(
            [row_of_slice[int(c)] for c in cand if int(c) in row_of_slice],
            dtype=np.int64,
        )
        if len(cand_rows) < 2:
            continue

        # Correlation window — exclude dtime zone for last slices
        if slice_info["is_last"][slice_idx]:
            win = slice(0, max(seg_len - dtime_samp_up, 1))
        else:
            win = slice(0, seg_len)

        target = aligned_segments[row].astype(np.float64)
        cand_data = aligned_segments[cand_rows].astype(np.float64)
        corr = corr_with_matrix(target[win], cand_data[:, win])
        order = np.argsort(corr)[::-1]
        keep_rows = cand_rows[order[: min(n_candidates, len(cand_rows))]]

        template = aligned_segments[keep_rows].mean(axis=0).astype(np.float64)
        denom = float(np.dot(template[win], template[win]))
        if denom < 1e-12:
            continue
        scaling = float(np.dot(target[win], template[win]) / denom)
        artifact_segments[row] = (scaling * template).astype(np.float32)

    return artifact_segments