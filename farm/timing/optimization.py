"""Global timing optimisation and slice-marker computation.

Key design principle
--------------------
Every volume is anchored to its real scanner trigger. The optimisation adjusts
the intra-volume slice-group duration only; it never lets timing drift away
from later volume triggers.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar

from farm.alignment.phase_shift import extract_aligned_segments
from farm.utils.signal import standardize_rows
from farm.utils.slices import build_slice_info

logger = logging.getLogger("farm.timing.optimization")


def compute_slice_markers(
    vol_onsets_up: np.ndarray,
    sdur: float,
    dtime: float,
    srate_up: float,
    n_sg: int,
    n_vol: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute rounded and exact slice-group onset markers.

    Real volume trigger locations are used as anchors. ``dtime`` is retained
    in the public API for compatibility and diagnostics, but does not shift
    volume locations: scanner triggers are more trustworthy than a theoretical
    accumulated timing model.

    Parameters
    ----------
    vol_onsets_up
        Real volume onset samples, at upsampled rate.
    sdur
        Slice-group duration in seconds.
    dtime
        Inter-volume dead-time in seconds. Informative only here.
    srate_up
        Upsampled sampling rate.
    n_sg
        Number of acquisition slice-groups per volume.
    n_vol
        Number of volumes. Defaults to len(vol_onsets_up).

    Returns
    -------
    onsets_up
        Rounded slice-group onset samples.
    round_errors
        Exact onset minus rounded onset, in samples.
    seg_len
        Rounded slice-group length in samples.
    """
    del dtime  # Kept only for backward-compatible API.

    vol_onsets_up = np.asarray(vol_onsets_up, dtype=np.int64)

    if n_vol is None:
        n_vol = len(vol_onsets_up)

    if n_vol < 1:
        raise ValueError("n_vol must be >= 1")

    if len(vol_onsets_up) < n_vol:
        raise ValueError("vol_onsets_up has fewer entries than n_vol")

    if sdur <= 0:
        raise ValueError(f"sdur must be positive, got {sdur}")

    if n_sg < 1:
        raise ValueError("n_sg must be >= 1")

    seg_len = int(round(sdur * srate_up))
    if seg_len < 2:
        raise ValueError(
            f"Slice segment too short ({seg_len} samples); check timing."
        )

    exact = np.empty(n_vol * n_sg, dtype=np.float64)

    for v in range(n_vol):
        base = float(vol_onsets_up[v])
        group_idx = np.arange(n_sg, dtype=np.float64)
        exact[v * n_sg:(v + 1) * n_sg] = (
            base + group_idx * sdur * srate_up
        )

    onsets_up = np.rint(exact).astype(np.int64)
    round_errors = exact - onsets_up.astype(np.float64)

    return onsets_up, round_errors, seg_len


def _global_cost_sdur(
    sdur: float,
    signal_ref: np.ndarray,
    vol_onsets_up: np.ndarray,
    n_sg: int,
    n_volumes: int,
    slice_meta: dict,
    srate_up: float,
    padding: int,
) -> float:
    """Timing cost for one candidate intra-volume slice duration."""
    if sdur <= 0:
        return 1e20

    seg_len = int(round(sdur * srate_up))
    if seg_len < 8:
        return 1e20

    # Require room for at least the acquisition train in almost all volumes.
    intervals = np.diff(vol_onsets_up[:n_volumes]).astype(np.float64)
    if len(intervals) > 0:
        min_interval = np.percentile(intervals, 5)
        if n_sg * seg_len > min_interval:
            return 1e20

    onsets, round_errors, _ = compute_slice_markers(
        vol_onsets_up=vol_onsets_up,
        sdur=sdur,
        dtime=0.0,
        srate_up=srate_up,
        n_sg=n_sg,
        n_vol=n_volumes,
    )

    segs, valid_idx = extract_aligned_segments(
        signal_ref,
        onsets,
        seg_len,
        round_errors,
        padding=padding,
        indices=slice_meta["good_slice_idx"],
    )

    if len(valid_idx) < max(10, n_sg):
        return 1e20

    z = standardize_rows(segs)
    return float(np.mean(np.std(z, axis=0)))


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
) -> tuple[float, float, object]:
    """Refine intra-volume slice duration while preserving real trigger timing.

    ``sdur`` is optimized as a scalar. ``dtime`` is then derived from the
    measured inter-trigger interval:

        dtime = median(TR_observed) - n_sg * sdur

    This avoids the old behavior where small timing errors accumulated over
    many volumes because only the first trigger was used as an anchor.
    """
    del dtime_init  # Initial dtime is not independently optimized anymore.

    ref_signal_up = np.asarray(ref_signal_up)
    vol_onsets_up = np.asarray(vol_onsets_up, dtype=np.int64)

    if n_vol < 2:
        raise ValueError("Need at least two volumes for timing optimisation")

    if len(vol_onsets_up) < n_vol:
        raise ValueError("vol_onsets_up has fewer entries than n_vol")

    if n_sg < 2:
        raise ValueError("n_sg must be >= 2")

    observed_intervals = np.diff(vol_onsets_up[:n_vol]).astype(np.float64)
    tr_samples_median = float(np.median(observed_intervals))
    tr_seconds_median = tr_samples_median / srate_up

    # A small safety margin avoids allowing the acquisition train to spill
    # into the next trigger interval.
    max_sdur = 0.999 * tr_seconds_median / n_sg

    # Allow a reasonable local search around the initial estimate, while
    # ensuring a broad enough interval to recover from poor coarse estimates.
    lower = max(8.0 / srate_up, 0.80 * sdur_init)
    upper = min(max_sdur, 1.20 * sdur_init)

    if lower >= upper:
        lower = max(8.0 / srate_up, 0.50 * max_sdur)
        upper = max_sdur

    slice_meta = build_slice_info(
        n_total=n_vol * n_sg,
        n_sg=n_sg,
        window_size=window_size,
    )

    result = minimize_scalar(
        _global_cost_sdur,
        bounds=(lower, upper),
        method="bounded",
        args=(
            ref_signal_up,
            vol_onsets_up,
            n_sg,
            n_vol,
            slice_meta,
            srate_up,
            padding,
        ),
        options={
            "xatol": 1e-10,
            "maxiter": 300,
        },
    )

    sdur = float(result.x)
    dtime = max(0.0, tr_seconds_median - n_sg * sdur)

    logger.info(
        "Optimised timing: sdur=%.6f ms, dtime=%.6f ms, "
        "observed TR=%.6f ms, cost=%.6f, success=%s",
        sdur * 1e3,
        dtime * 1e3,
        tr_seconds_median * 1e3,
        float(result.fun),
        bool(result.success),
    )

    return sdur, dtime, result