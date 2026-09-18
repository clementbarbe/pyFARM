"""Slice-group metadata and template-candidate construction."""

import numpy as np


def build_slice_info(
    n_total: int,
    n_sg: int,
    window_size: int,
) -> dict:
    """Build slice-group metadata.

    Candidate templates are restricted to the same acquisition slice-group
    across nearby volumes. This is preferable to mixing different groups
    within an EPI volume, whose artifact morphology can differ substantially.

    Parameters
    ----------
    n_total
        Total number of slice-group segments.
    n_sg
        Number of acquisition slice-groups per fMRI volume.
    window_size
        Half-width of local candidate search, expressed in VOLUMES.

        For example, ``window_size=50`` searches candidate segments from
        up to 50 previous and 50 following volumes, always in the same
        slice-group.

    Returns
    -------
    dict
        marker_vector, volume_index, group_index, is_first, is_last,
        good_slice_idx, last_slice_idx, candidate_idx.
    """
    if n_total < 1:
        raise ValueError("n_total must be >= 1")

    if n_sg < 1:
        raise ValueError("n_sg must be >= 1")

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    marker_vector = np.arange(n_total, dtype=np.int64)
    volume_index = marker_vector // n_sg
    group_index = marker_vector % n_sg
    n_volumes = int(np.ceil(n_total / n_sg))

    is_first = group_index == 0
    is_last = group_index == (n_sg - 1)

    # Used for timing-cost estimation. Boundary groups may be contaminated
    # by transitions between volumes, so they are excluded from that cost.
    good = ~(is_first | is_last)
    good_idx = np.flatnonzero(good).astype(np.int64)

    candidate_lists: list[np.ndarray] = []

    for idx in marker_vector:
        vol = int(volume_index[idx])
        group = int(group_index[idx])

        v_start = max(0, vol - window_size)
        v_stop = min(n_volumes, vol + window_size + 1)

        candidate_volumes = np.arange(v_start, v_stop, dtype=np.int64)
        candidates = candidate_volumes * n_sg + group

        # Last partial volume can generate an index beyond n_total.
        candidates = candidates[candidates < n_total]

        # Exclude self.
        candidates = candidates[candidates != idx]

        # Sort nearest temporal volumes first. The template function will
        # subsequently retain the n_candidates best-correlated segments.
        order = np.argsort(np.abs((candidates // n_sg) - vol))
        candidates = candidates[order].astype(np.int64)

        candidate_lists.append(candidates)

    max_candidates = max((len(x) for x in candidate_lists), default=0)
    candidate_idx = -np.ones((n_total, max_candidates), dtype=np.int64)

    for idx, candidates in enumerate(candidate_lists):
        candidate_idx[idx, :len(candidates)] = candidates

    return {
        "marker_vector": marker_vector,
        "volume_index": volume_index,
        "group_index": group_index,
        "is_first": is_first,
        "is_last": is_last,
        "good_slice_idx": good_idx,
        "last_slice_idx": np.flatnonzero(is_last).astype(np.int64),
        "candidate_idx": candidate_idx,
    }