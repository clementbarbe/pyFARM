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
    max_interval_error_seconds: float = 0.005,
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

    def marker_label(text):
        return "".join(text.rsplit("/", 1)[-1].split()).casefold()

    matches = [val for key, val in event_dict.items()
               if key.strip().casefold() == trigger.strip().casefold()]
    if not matches:
        matches = [val for key, val in event_dict.items()
                   if marker_label(key) == marker_label(trigger)]
    if not matches:
        raise ValueError(
            f"Trigger '{trigger}' not found. Available: {list(event_dict.keys())}"
        )
    if len(matches) != 1:
        raise ValueError(f"Ambiguous trigger '{trigger}': multiple event types match")
    target_id = matches[0]

    srate = float(raw.info["sfreq"])
    vol_onsets = events[events[:, 2] == target_id, 0].astype(np.int64) - raw.first_samp

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
    tolerance = max(float(max_interval_error_seconds), 2.0 / srate)
    if np.any(ivis <= 0) or np.any(np.abs(ivis - tr) > tolerance):
        raise ValueError(
            f"Irregular volume triggers: interval range {ivis.min():.6f}–{ivis.max():.6f} s, "
            f"expected TR={tr:.6f} s +/-{tolerance:.6f} s. "
            "Check missing/duplicate triggers, acquisition pauses, and configured TR."
        )
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
