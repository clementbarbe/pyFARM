"""Global Nelder-Mead optimisation of sdur / dtime and
slice-marker computation."""

import logging

import numpy as np
from scipy.optimize import minimize

from farm.alignment.phase_shift import extract_aligned_segments
from farm.utils.signal import standardize_rows
from farm.utils.slices import build_slice_info

logger = logging.getLogger("farm.timing.optimization")


# ── Slice-marker geometry ───────────────────────────────────────

def compute_slice_markers(
    vol_onsets_up: np.ndarray,
    sdur: float,
    dtime: float,
    srate_up: float,
    n_sg: int,
    n_vol: int,
) -> tuple:
    """Compute exact onset of every slice segment, then round.

    Returns
    -------
    onsets_up : int64 array, length ``n_vol * n_sg``.
    round_errors : float64 array, same length.
    seg_len : int — segment length in upsampled samples.
    """
    n_total = n_vol * n_sg
    vol0 = float(vol_onsets_up[0])
    exact = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        v = i // n_sg
        exact[i] = vol0 + (i * sdur + v * dtime) * srate_up
    onsets_up = np.rint(exact).astype(np.int64)
    round_errors = exact - onsets_up.astype(np.float64)
    seg_len = int(round(sdur * srate_up))
    return onsets_up, round_errors, seg_len


# ── Cost function ───────────────────────────────────────────────

def _global_cost(
    params,
    signal_ref,
    onset_first_up,
    n_sg,
    n_volumes,
    slice_meta,
    srate_up,
    padding,
):
    sdur, dtime = float(params[0]), float(params[1])
    if sdur <= 0 or dtime < 0:
        return 1e20
    seg_len_loc = int(round(sdur * srate_up))
    if seg_len_loc < 8:
        return 1e20

    n_total = n_sg * n_volumes
    exact = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        v = i // n_sg
        exact[i] = onset_first_up + (i * sdur + v * dtime) * srate_up
    rounded = np.rint(exact).astype(np.int64)
    round_err = exact - rounded.astype(np.float64)

    segs, valid_idx = extract_aligned_segments(
        signal_ref, rounded, seg_len_loc, round_err,
        padding=padding,
        indices=slice_meta["good_slice_idx"],
    )
    if len(valid_idx) < max(10, n_sg):
        return 1e20
    z = standardize_rows(segs)
    return float(np.mean(np.std(z, axis=0)))


# ── Public API ──────────────────────────────────────────────────

def optimize_global_timing(
    ref_signal_up: np.ndarray,
    srate_up: float,
    vol_onsets_up: np.ndarray,
    sdur_init: float,
    dtime_init: float,
    n_sg: int,
    n_vol: int,
    padding: int = 10,
    window_size: int = 50,
) -> tuple:
    """Nelder-Mead refinement of *sdur* and *dtime*.

    Parameters
    ----------
    ref_signal_up : 1-D array — upsampled reference channel.
    srate_up : float — upsampled sampling rate.
    vol_onsets_up : int64 array.
    sdur_init, dtime_init : float — initial estimates (s).
    n_sg, n_vol : int.
    padding : int — samples for FFT phase-shift.
    window_size : int — candidate window for slice info.

    Returns
    -------
    sdur : float — optimised slice duration (s).
    dtime : float — optimised dead-time (s).
    result : ``scipy.optimize.OptimizeResult``.
    """
    n_total = n_vol * n_sg
    slice_meta = build_slice_info(n_total, n_sg, window_size)

    result = minimize(
        _global_cost,
        x0=[sdur_init, dtime_init],
        args=(
            ref_signal_up,
            float(vol_onsets_up[0]),
            n_sg,
            n_vol,
            slice_meta,
            srate_up,
            padding,
        ),
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-6,
                 "maxiter": 600, "adaptive": True},
    )

    sdur = float(result.x[0])
    dtime = float(result.x[1])
    logger.info(
        "Optimised: sdur=%.6f ms, dtime=%.6f ms (cost=%.6f, iter=%d, ok=%s)",
        sdur * 1e3, dtime * 1e3, result.fun, result.nit, result.success,
    )
    return sdur, dtime, result