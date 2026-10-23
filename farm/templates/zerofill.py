"""Inter-volume dead-time masking (FARM step iv)."""

import numpy as np


def gap_mask_intervals(
    onsets: np.ndarray,
    last_slice_idx: np.ndarray,
    seg_len: int,
    gap_fraction: float,
    *,
    segment_stops: np.ndarray | None = None,
    scan_stop: int | None = None,
) -> list[list[int]]:
    """Return half-open sample intervals explicitly removed by gap masking."""
    if not 0.0 <= gap_fraction <= 1.0:
        raise ValueError("gap_fraction must be in [0, 1]")
    intervals = []
    if gap_fraction == 0:
        return intervals
    for idx in np.asarray(last_slice_idx, dtype=np.int64):
        if idx < 0 or idx >= len(onsets):
            continue
        start = int(segment_stops[idx]) if segment_stops is not None else int(onsets[idx]) + seg_len
        if idx + 1 < len(onsets):
            end = int(onsets[idx + 1])
        elif scan_stop is not None:
            end = int(scan_stop)
        else:
            continue
        stop = start + int(round(gap_fraction * max(0, end - start)))
        if stop > start:
            intervals.append([start, stop])
    return intervals


def zero_fill_dtime(
    signal: np.ndarray,
    onsets_up: np.ndarray,
    last_slice_idx: np.ndarray,
    seg_len: int,
    dtime_samp_up: int | None = None,
    gap_fraction: float = 1.0,
    *,
    segment_stops: np.ndarray | None = None,
    scan_stop: int | None = None,
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

    for start, stop in gap_mask_intervals(
        onsets_up, last_slice_idx, seg_len, gap_fraction,
        segment_stops=segment_stops, scan_stop=scan_stop,
    ):
        start, stop = max(0, start), min(stop, len(out))
        if stop > start:
            out[start:stop] = 0.0

    return out
