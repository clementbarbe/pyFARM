"""Quantitative metrics for denoising quality."""

import logging

import numpy as np
from farm.preprocessing.filters import apply_bandpass

logger = logging.getLogger("farm.diagnostics.metrics")


def compute_rms_reduction(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_names: list,
    bandpass: tuple | None = (30, 250),
) -> list:
    """Compute per-channel RMS before/after and the reduction ratio.

    Returns
    -------
    list of dict — one per channel, with keys
    ``ch_name``, ``rms_before``, ``rms_after``, ``ratio``.
    """
    results = []
    for i, name in enumerate(ch_names):
        ts_b = apply_bandpass(data_before[i], srate, bandpass)
        ts_a = apply_bandpass(data_after[i], srate, bandpass)
        rms_b = float(np.sqrt(np.mean(ts_b ** 2)))
        rms_a = float(np.sqrt(np.mean(ts_a ** 2)))
        ratio = rms_b / max(rms_a, 1e-20)
        status = "✅" if ratio > 3 else ("⚠️" if ratio > 1.5 else "❌")
        logger.info(
            "  %s %s  RMS: %.2e → %.2e  (reduction %.1f×)",
            status, name, rms_b, rms_a, ratio,
        )
        results.append(dict(
            ch_name=name, rms_before=rms_b, rms_after=rms_a, ratio=ratio,
        ))
    return results