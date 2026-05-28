"""Slice-group metadata builder."""

import numpy as np


def build_slice_info(n_total: int, n_sg: int, window_size: int) -> dict:
    """Build metadata describing which slices are first/last per volume,
    and which candidates are available for template building.

    Parameters
    ----------
    n_total : int
        Total number of slice segments across all volumes.
    n_sg : int
        Number of slice groups per volume.
    window_size : int
        Maximum number of candidate slices considered per segment.

    Returns
    -------
    dict with keys:
        marker_vector, is_first, is_last,
        good_slice_idx, last_slice_idx, candidate_idx
    """
    marker_vector = np.arange(n_total, dtype=np.int64)

    is_last = ((marker_vector + 1) % n_sg) == 0
    is_first = np.zeros(n_total, dtype=bool)
    is_first[::n_sg] = True

    good = ~(is_first | is_last)
    good_idx = np.flatnonzero(good)

    n_keep = min(window_size, max(1, len(good_idx) - 1))
    candidate_idx = -np.ones((n_total, n_keep), dtype=np.int64)
    for i in range(n_total):
        others = good_idx[good_idx != i]
        order = np.argsort(np.abs(others - i))
        picked = np.sort(others[order[:n_keep]])
        candidate_idx[i, : len(picked)] = picked

    return {
        "marker_vector": marker_vector,
        "is_first": is_first,
        "is_last": is_last,
        "good_slice_idx": good_idx,
        "last_slice_idx": np.flatnonzero(is_last),
        "candidate_idx": candidate_idx,
    }