"""Trim the working data to the scan boundaries."""

import logging
import numpy as np

logger = logging.getLogger("farm.preprocessing.trim")


def trim_to_scan(
    data: np.ndarray,
    vol_onsets: np.ndarray,
    srate: float,
    tr: float,
) -> tuple:
    """Crop the data array to the region covered by the scan.

    This is an *internal* operation — the full-length signal is
    reconstructed before export so that no data are lost.

    Parameters
    ----------
    data : ndarray, shape ``(n_ch, n_samples_full)``.
    vol_onsets : 1-D int64 array — absolute volume-onset samples.
    srate : float
    tr : float — repetition time (s).

    Returns
    -------
    data_crop : ndarray, cropped copy.
    vol_onsets_crop : int64 array — onsets relative to the crop start.
    vol_onsets_abs : int64 array — original absolute onsets (unchanged).
    s_trim : int — start sample of the crop in the original signal.
    e_trim : int — end sample of the crop in the original signal.
    """
    s_trim = max(0, int(vol_onsets[0]))
    e_trim = min(data.shape[1], int(vol_onsets[-1] + tr * srate))

    data_crop = data[:, s_trim:e_trim].copy()
    vol_onsets_abs = vol_onsets.copy()
    vol_onsets_crop = (vol_onsets - s_trim).astype(np.int64)

    n_removed = data.shape[1] - data_crop.shape[1]
    logger.info(
        "Trim: samples %d–%d kept (%d removed, %.1f s working region)",
        s_trim, e_trim, n_removed, data_crop.shape[1] / srate,
    )
    return data_crop, vol_onsets_crop, vol_onsets_abs, s_trim, e_trim