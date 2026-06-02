"""Trim the working data to the scan boundaries.

After trimming, a second pass verifies that the last retained volume
still has a full TR of data.  If not it is dropped so that every
volume in the returned arrays is guaranteed complete.
"""

import logging
import numpy as np

logger = logging.getLogger("farm.preprocessing.trim")


def trim_to_scan(
    data: np.ndarray,
    vol_onsets: np.ndarray,
    srate: float,
    tr: float,
) -> tuple:
    """Crop the data array to the region covered by complete volumes.

    This is an *internal* operation — the full-length signal is
    reconstructed before export so that no data are lost.

    Any trailing volume whose onset + TR would fall outside the
    cropped region is automatically removed.

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
    vol_onsets_abs : int64 array — original absolute onsets (complete only).
    s_trim : int — start sample of the crop in the original signal.
    e_trim : int — end sample of the crop in the original signal.
    n_vol : int — number of complete volumes after trimming.
    """
    tr_samples = int(np.ceil(tr * srate))
    n_total_samples = data.shape[1]

    # ── First pass: compute crop boundaries ──────────────────
    s_trim = max(0, int(vol_onsets[0]))
    e_trim = min(n_total_samples, int(vol_onsets[-1]) + tr_samples)

    # ── Second pass: drop any volume that overflows e_trim ───
    n_before = len(vol_onsets)
    while len(vol_onsets) > 0:
        last_end = int(vol_onsets[-1]) + tr_samples
        if last_end <= e_trim:
            break
        vol_onsets = vol_onsets[:-1]
        # Recompute e_trim to the new last volume
        if len(vol_onsets) > 0:
            e_trim = min(n_total_samples, int(vol_onsets[-1]) + tr_samples)

    n_removed = n_before - len(vol_onsets)
    if n_removed > 0:
        logger.warning(
            "Removed %d incomplete volume(s) during trim "
            "(need %d samples after each onset).",
            n_removed, tr_samples,
        )

    if len(vol_onsets) < 2:
        raise ValueError(
            f"Only {len(vol_onsets)} complete volume(s) after trim — "
            f"need at least 2."
        )

    # ── Crop ─────────────────────────────────────────────────
    data_crop = data[:, s_trim:e_trim].copy()
    vol_onsets_abs = vol_onsets.copy()
    vol_onsets_crop = (vol_onsets - s_trim).astype(np.int64)
    n_vol = len(vol_onsets_crop)

    logger.info(
        "Trim: samples %d–%d kept (%d complete volumes, %.1f s working region)",
        s_trim, e_trim, n_vol, data_crop.shape[1] / srate,
    )
    return data_crop, vol_onsets_crop, vol_onsets_abs, s_trim, e_trim, n_vol