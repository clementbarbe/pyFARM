"""Volume-trigger detection and validation."""

import logging

import numpy as np
import mne

logger = logging.getLogger("farm.io.triggers")


def detect_volume_onsets(
    raw: mne.io.Raw,
    trigger: str,
    tr: float,
    n_volumes: int | None = None,
    drop_last: bool = False,
) -> tuple:
    """Extract volume-onset sample indices from annotations.

    Parameters
    ----------
    raw : MNE Raw object.
    trigger : str — marker label to search for.
    tr : float — expected TR (s), used for validation.
    n_volumes : int or None — keep only the first *n* volumes.
    drop_last : bool — if *True*, unconditionally remove the last
        detected volume.  Use when the acquisition was stopped
        manually and the final volume may be incomplete.

    Returns
    -------
    vol_onsets : int64 array — sample indices of volume onsets.
    n_vol : int — number of retained volumes.
    ivi_stats : dict — inter-volume-interval statistics.
    """
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    logger.debug("Event types found: %s", event_dict)

    target_id = None
    for key, val in event_dict.items():
        if trigger in key:
            target_id = val
            break
    if target_id is None:
        raise ValueError(
            f"Trigger '{trigger}' not found. Available: {list(event_dict.keys())}"
        )

    srate = float(raw.info["sfreq"])
    vol_onsets = events[events[:, 2] == target_id, 0].astype(np.int64)

    if n_volumes is not None:
        vol_onsets = vol_onsets[:n_volumes]

    # ── Drop last volume if requested (manual stop) ──────────
    if drop_last and len(vol_onsets) > 1:
        logger.info(
            "Dropping last volume (onset sample %d, t=%.2f s) — "
            "manual acquisition stop assumed.",
            vol_onsets[-1], vol_onsets[-1] / srate,
        )
        vol_onsets = vol_onsets[:-1]

    if len(vol_onsets) < 2:
        raise ValueError(
            f"Only {len(vol_onsets)} volume(s) found — "
            f"need at least 2 to run the pipeline."
        )

    n_vol = len(vol_onsets)

    # ── IVI statistics ───────────────────────────────────────
    ivis = np.diff(vol_onsets) / srate
    ivi_stats = {
        "mean": float(np.mean(ivis)),
        "std": float(np.std(ivis)),
        "jitter_max": float(np.max(np.abs(ivis - tr))),
        "ivis": ivis,
    }

    logger.info(
        "Detected %d volumes (IVI mean=%.4f s, jitter_max=%.1f µs)",
        n_vol, ivi_stats["mean"], ivi_stats["jitter_max"] * 1e6,
    )
    return vol_onsets, n_vol, ivi_stats