"""Coarse per-volume estimation of sdur and dtime (PREPROCESSING stage)."""

import logging

import numpy as np
from farm.preprocessing.filters import hpf_butter_1d

logger = logging.getLogger("farm.timing.estimation")


def _vol_variance(
    signal_1d: np.ndarray,
    vol_start: int,
    vol_dur: int,
    dtime_samp: int,
    n_sg: int,
) -> float:
    """Sum of per-sample standard deviations across slice segments
    within a single volume, for a given *dtime_samp*."""
    remaining = vol_dur - dtime_samp
    if remaining <= 0 or n_sg < 2:
        return np.inf
    sdur_f = remaining / n_sg
    seg_len = int(round(sdur_f))
    if seg_len < 8:
        return np.inf
    segs = np.empty((n_sg, seg_len), dtype=np.float64)
    for k in range(n_sg):
        start = int(round(vol_start + k * sdur_f))
        stop = start + seg_len
        if start < 0 or stop > len(signal_1d):
            return np.inf
        segs[k] = signal_1d[start:stop]
    return float(np.sum(np.std(segs, axis=0)))


def estimate_initial_timing(
    ref_signal: np.ndarray,
    srate: float,
    vol_onsets: np.ndarray,
    n_sg: int,
    n_vol: int,
) -> tuple:
    """Estimate initial *sdur* and *dtime* by scanning each volume.

    A 250 Hz high-pass is applied to the reference channel to
    isolate the high-frequency gradient artifact, following the
    MATLAB FARM implementation.

    Parameters
    ----------
    ref_signal : 1-D array — reference EMG channel (already HPF 30 Hz).
    srate : float
    vol_onsets : 1-D int array — volume onsets (samples).
    n_sg : int — number of slice groups per volume.
    n_vol : int — number of volumes.

    Returns
    -------
    sdur_init : float — initial slice duration (s).
    dtime_init : float — initial dead-time (s).
    diag : dict — per-volume values for optional plotting.
    """
    ref_250 = hpf_butter_1d(ref_signal, srate, 250.0)
    n_sample_per_tr = int(np.median(np.diff(vol_onsets)))
    dtime_max = int(round(n_sample_per_tr / n_sg / 2))
    dtime_candidates = np.arange(dtime_max + 1, dtype=np.int64)

    sdur_list, dtime_list, sv_list = [], [], []

    for v in range(n_vol):
        vol_start = int(vol_onsets[v])
        vol_dur = (
            int(vol_onsets[v + 1] - vol_onsets[v])
            if v + 1 < n_vol
            else n_sample_per_tr
        )
        best_sv, best_dt = np.inf, 0
        for dt in dtime_candidates:
            if int(dt) >= vol_dur:
                break
            sv = _vol_variance(ref_250, vol_start, vol_dur, int(dt), n_sg)
            if sv < best_sv:
                best_sv, best_dt = sv, int(dt)
        sdur_list.append((vol_dur - best_dt) / n_sg / srate)
        dtime_list.append(best_dt / srate)
        sv_list.append(best_sv)

    sdur_init = float(np.mean(sdur_list))
    dtime_init = float(np.mean(dtime_list))

    logger.info(
        "Initial timing: sdur=%.4f ms, dtime=%.4f ms",
        sdur_init * 1e3, dtime_init * 1e3,
    )

    diag = {
        "sdur_list": np.array(sdur_list),
        "dtime_list": np.array(dtime_list),
        "sv_list": np.array(sv_list),
        "dtime_candidates": dtime_candidates,
        "ref_250": ref_250,
        "n_sample_per_tr": n_sample_per_tr,
    }
    return sdur_init, dtime_init, diag