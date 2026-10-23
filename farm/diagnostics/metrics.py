"""Quantitative metrics for denoising quality."""

import logging

import numpy as np
from scipy import signal as sig
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
        logger.info(
            "  %s RMS: %.2e → %.2e (reduction %.1f×; not a signal-preservation score)",
            name, rms_b, rms_a, ratio,
        )
        results.append(dict(
            ch_name=name, rms_before=rms_b, rms_after=rms_a, ratio=ratio,
        ))
    return results


def compute_gradient_harmonic_reduction(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_names: list,
    sdur: float,
    band: tuple[float, float] = (30.0, 250.0),
) -> list:
    """Measure excess power around group harmonics above a local PSD floor.

    The local floor distinguishes narrow scanner peaks from broadband energy.
    This remains an artifact metric, not proof of physiological preservation.
    """
    nperseg = min(data_before.shape[-1], int(round(8 * srate)))
    if nperseg < 16:
        return []
    frequency = 1.0 / sdur
    results = []
    for i, name in enumerate(ch_names):
        f, pb = sig.welch(data_before[i].astype(np.float64), fs=srate, nperseg=nperseg)
        _, pa = sig.welch(data_after[i].astype(np.float64), fs=srate, nperseg=nperseg)
        df = float(f[1]-f[0])
        half_width = max(df, 0.25)
        centers = np.arange(max(1, int(np.ceil(band[0]/frequency))),
                            int(np.floor(band[1]/frequency))+1) * frequency
        excess = [0.0, 0.0]
        for center in centers:
            peak = (np.abs(f-center) <= half_width) & (f >= band[0]) & (f <= band[1])
            floor = (np.abs(f-center) > 2*half_width) & (np.abs(f-center) <= min(3.0, frequency/3))
            if not np.any(peak) or not np.any(floor):
                continue
            for j, psd in enumerate((pb, pa)):
                excess[j] += float(np.maximum(psd[peak]-np.median(psd[floor]), 0).sum()*df)
        ratio = excess[0] / excess[1] if excess[1] > 0 else np.nan
        results.append({"ch_name": name, "group_frequency_hz": frequency,
                        "excess_power_before": excess[0], "excess_power_after": excess[1],
                        "excess_power_ratio": ratio})
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
