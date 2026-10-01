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

def compute_scanner_locked_reduction(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_names: list,
    vol_onsets: np.ndarray,
    tr: float,
    bandpass: tuple | None = (30, 250),
) -> list:
    """Quantify trigger-locked artifact separately from global RMS.

    For each channel, full-TR epochs are aligned to volume triggers and their
    sample-wise median is used as the scanner-locked waveform.  The metric is
    the RMS of this locked waveform before/after cleaning.  A large reduction
    here is stronger evidence of scanner-artifact suppression than a global
    RMS decrease alone.
    """
    vol_onsets = np.asarray(vol_onsets, dtype=np.int64)
    tr_samples = max(8, int(round(float(tr) * float(srate))))
    results = []

    for i, name in enumerate(ch_names):
        before = apply_bandpass(data_before[i], srate, bandpass)
        after = apply_bandpass(data_after[i], srate, bandpass)

        epochs_b = []
        epochs_a = []
        for onset in vol_onsets:
            start = int(onset)
            stop = start + tr_samples
            if start >= 0 and stop <= len(before) and stop <= len(after):
                epochs_b.append(before[start:stop])
                epochs_a.append(after[start:stop])

        if len(epochs_b) < 3:
            results.append(dict(
                ch_name=name, locked_rms_before=np.nan,
                locked_rms_after=np.nan, locked_ratio=np.nan,
            ))
            continue

        Xb = np.asarray(epochs_b, dtype=np.float64)
        Xa = np.asarray(epochs_a, dtype=np.float64)
        locked_b = np.median(Xb, axis=0)
        locked_a = np.median(Xa, axis=0)
        rms_b = float(np.sqrt(np.mean(locked_b * locked_b)))
        rms_a = float(np.sqrt(np.mean(locked_a * locked_a)))
        ratio = rms_b / max(rms_a, 1e-20)
        logger.info(
            "  %s scanner-locked RMS: %.2e → %.2e (reduction %.1f×)",
            name, rms_b, rms_a, ratio,
        )
        results.append(dict(
            ch_name=name,
            locked_rms_before=rms_b,
            locked_rms_after=rms_a,
            locked_ratio=ratio,
        ))

    return results
