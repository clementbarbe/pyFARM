"""Zero-fill the dead-time (dtime) windows around volume boundaries
(FARM step iv)."""

import numpy as np


def zero_fill_dtime(
    signal: np.ndarray,
    onsets_up: np.ndarray,
    last_slice_idx: np.ndarray,
    seg_len: int,
    dtime_samp_up: int,
) -> np.ndarray:
    """Set the inter-volume dead-time regions to zero.

    Parameters
    ----------
    signal : 1-D array (modified in-place on a copy).
    onsets_up : int64 array of all slice onsets (upsampled).
    last_slice_idx : int64 array — indices of last-of-volume slices.
    seg_len : int — slice segment length (upsampled samples).
    dtime_samp_up : int — dead-time in upsampled samples.

    Returns
    -------
    Copy of *signal* with zero-filled dtime windows.
    """
    out = signal.copy()
    for last_idx in last_slice_idx:
        start = int(onsets_up[last_idx] + seg_len - dtime_samp_up)
        stop = int(onsets_up[last_idx] + seg_len + dtime_samp_up)
        start = max(0, min(start, len(out)))
        stop = max(0, min(stop, len(out)))
        if stop > start:
            out[start:stop] = 0.0
    return out