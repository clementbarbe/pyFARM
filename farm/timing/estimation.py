"""Coarse per-volume estimation of sdur and dtime (PREPROCESSING stage)."""

import logging

import numpy as np
from farm.preprocessing.filters import hpf_butter_1d
from farm.utils.signal import standardize_rows

logger = logging.getLogger("farm.timing.estimation")


def _vol_variance(
    signal_1d: np.ndarray,
    vol_start: int,
    vol_dur: int,
    dtime_samp: int,
    n_sg: int,
) -> float:
    """Normalized mean deviation across interior groups for one gap candidate."""
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
    # A sum rewards shorter segments even for unrelated data. Compare equal
    # units and ignore boundary groups when enough interior groups exist.
    if n_sg > 3:
        segs = segs[1:-1]
    return float(np.mean(np.std(standardize_rows(segs), axis=0)))


def estimate_initial_timing(
    ref_signal: np.ndarray,
    srate: float,
    vol_onsets: np.ndarray,
    n_sg: int,
    n_vol: int,
    *,
    max_gap_seconds: float = 0.2,
    highpass_cutoff: float = 250.0,
) -> tuple:
    """Estimate initial *sdur* and *dtime* by scanning each volume.

    A configurable high-pass (250 Hz by default) isolates the timing reference.
    Its band must remain scanner-dominated after acquisition filtering.

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
    if not (0.0 < highpass_cutoff < srate / 2):
        raise ValueError("Timing high-pass must be between zero and Nyquist")
    if n_sg < 2 or n_vol < 2 or len(vol_onsets) < n_vol:
        raise ValueError("Timing requires at least two groups and two volume onsets")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    ref_250 = hpf_butter_1d(ref_signal, srate, highpass_cutoff)
    n_sample_per_tr = int(np.median(np.diff(vol_onsets)))
    dtime_max = min(int(round(max_gap_seconds * srate)), n_sample_per_tr - 8 * n_sg)
    if dtime_max < 0:
        raise ValueError("TR is too short for the requested number of groups")
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
        if not np.isfinite(best_sv):
            continue
        sdur_list.append((vol_dur - best_dt) / n_sg / srate)
        dtime_list.append(best_dt / srate)
        sv_list.append(best_sv)

    # Use robust medians rather than means. A few volumes can contain strong
    # voluntary EMG or transient motion; those volumes must not pull the
    # sequence timing away from the scanner's fixed acquisition pattern.
    if not sdur_list:
        raise ValueError("No complete valid volumes for initial timing estimation")
    sdur_arr = np.asarray(sdur_list, dtype=np.float64)
    dtime_arr = np.asarray(dtime_list, dtype=np.float64)
    sdur_init = float(np.median(sdur_arr))
    dtime_init = float(np.median(dtime_arr))

    sdur_mad = float(np.median(np.abs(sdur_arr - sdur_init)))
    dtime_mad = float(np.median(np.abs(dtime_arr - dtime_init)))

    logger.info(
        "Initial timing: sdur=%.4f ms, dtime=%.4f ms "
        "(volume MAD %.4f / %.4f ms)",
        sdur_init * 1e3, dtime_init * 1e3,
        sdur_mad * 1e3, dtime_mad * 1e3,
    )

    diag = {
        "sdur_list": np.array(sdur_list),
        "dtime_list": np.array(dtime_list),
        "sv_list": np.array(sv_list),
        "dtime_candidates": dtime_candidates,
        "ref_250": ref_250,
        "n_sample_per_tr": n_sample_per_tr,
        "sdur_mad": sdur_mad,
        "dtime_mad": dtime_mad,
        "search_max_gap_seconds": dtime_max / srate,
        "at_upper_bound": np.asarray(dtime_list) >= dtime_max / srate,
        "highpass_cutoff_hz": highpass_cutoff,
    }
    return sdur_init, dtime_init, diag
