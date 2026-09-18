"""Inter-volume dead-time masking (FARM step iv)."""

import numpy as np


def zero_fill_dtime(
    signal: np.ndarray,
    onsets_up: np.ndarray,
    last_slice_idx: np.ndarray,
    seg_len: int,
    dtime_samp_up: int | None = None,
    gap_fraction: float = 1.0,
) -> np.ndarray:
    """Zero-fill actual inter-volume gaps.

    The old implementation removed a symmetric region around the end of the
    final slice, including samples belonging to the final acquisition segment.

    This implementation masks only the actual gap:

        end_of_last_group --> onset_of_first_group_next_volume

    Parameters
    ----------
    signal
        One-dimensional signal.
    onsets_up
        All slice-group onsets.
    last_slice_idx
        Global indices of last slice-group in each volume.
    seg_len
        Slice-group segment length in samples.
    dtime_samp_up
        Retained only for backwards compatibility; not used.
    gap_fraction
        Fraction of the true inter-volume gap to set to zero.
        1.0 masks all of it; 0.0 disables masking.

    Returns
    -------
    ndarray
        Copy of the signal with selected inter-volume intervals set to zero.
    """
    del dtime_samp_up

    if not (0.0 <= gap_fraction <= 1.0):
        raise ValueError("gap_fraction must be in [0, 1]")

    out = np.asarray(signal).copy()
    onsets_up = np.asarray(onsets_up, dtype=np.int64)
    last_slice_idx = np.asarray(last_slice_idx, dtype=np.int64)

    if gap_fraction == 0.0:
        return out

    for last_idx in last_slice_idx:
        if last_idx < 0 or last_idx >= len(onsets_up):
            continue

        next_idx = int(last_idx) + 1

        # There is no known following volume boundary after the final volume.
        if next_idx >= len(onsets_up):
            continue

        last_segment_end = int(onsets_up[last_idx]) + int(seg_len)
        next_volume_start = int(onsets_up[next_idx])

        if next_volume_start <= last_segment_end:
            continue

        gap_len = next_volume_start - last_segment_end
        fill_stop = last_segment_end + int(round(gap_fraction * gap_len))

        start = max(0, min(last_segment_end, len(out)))
        stop = max(0, min(fill_stop, len(out)))

        if stop > start:
            out[start:stop] = 0.0

    return out