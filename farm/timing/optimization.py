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
    refine_half_width_seconds: float = 0.0005,
    min_relative_improvement: float = 0.005,
) -> tuple[float, float, object]:
    """Locally refine slice-group duration around a robust initial estimate.

    The EPI sequence timing is a scanner/protocol property. A denoising
    objective is highly periodic and therefore contains harmonic and
    sub-harmonic minima. A broad optimiser can converge perfectly to a
    numerically valid but physically wrong timing.

    This routine therefore *only* performs a small local refinement around
    ``sdur_init``. If the optimum hits the search boundary or does not
    improve the objective by ``min_relative_improvement``, the initial timing
    is retained. ``dtime`` is always derived from the measured TR.
    """
    del dtime_init

    ref_signal_up = np.asarray(ref_signal_up)
    vol_onsets_up = np.asarray(vol_onsets_up, dtype=np.int64)

    if n_vol < 2:
        raise ValueError("Need at least two volumes for timing optimisation")
    if len(vol_onsets_up) < n_vol:
        raise ValueError("vol_onsets_up has fewer entries than n_vol")
    if n_sg < 2:
        raise ValueError("n_sg must be >= 2")
    if refine_half_width_seconds <= 0:
        raise ValueError("refine_half_width_seconds must be > 0")
    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be >= 0")

    observed_intervals = np.diff(vol_onsets_up[:n_vol]).astype(np.float64)
    tr_samples_median = float(np.median(observed_intervals))
    tr_seconds_median = tr_samples_median / srate_up
    max_sdur = 0.999 * tr_seconds_median / n_sg

    lower = max(8.0 / srate_up, sdur_init - refine_half_width_seconds)
    upper = min(max_sdur, sdur_init + refine_half_width_seconds)
    if lower >= upper:
        raise ValueError(
            "No valid local timing interval around sdur_init; check TR/n_sg"
        )

    slice_meta = build_slice_info(
        n_total=n_vol * n_sg,
        n_sg=n_sg,
        window_size=window_size,
    )

    args = (
        ref_signal_up, vol_onsets_up, n_sg, n_vol, slice_meta, srate_up, padding
    )
    cost_init = _global_cost_sdur(sdur_init, *args)

    result = minimize_scalar(
        _global_cost_sdur,
        bounds=(lower, upper),
        method="bounded",
        args=args,
        options={"xatol": 1e-9, "maxiter": 120},
    )

    candidate = float(result.x)
    cost_candidate = float(result.fun)
    relative_improvement = (cost_init - cost_candidate) / (abs(cost_init) + 1e-30)

    width = upper - lower
    edge_tol = max(5.0 / srate_up, 0.02 * width)
    hits_edge = (candidate - lower <= edge_tol) or (upper - candidate <= edge_tol)

    accepted = (
        bool(result.success)
        and np.isfinite(cost_candidate)
        and not hits_edge
        and relative_improvement >= min_relative_improvement
    )

    if accepted:
        sdur = candidate
        decision = "accepted local refinement"
    else:
        sdur = float(sdur_init)
        decision = "kept robust initial timing"

    dtime = max(0.0, tr_seconds_median - n_sg * sdur)

    result.cost_initial = float(cost_init)
    result.relative_improvement = float(relative_improvement)
    result.hits_edge = bool(hits_edge)
    result.accepted = bool(accepted)
    result.sdur_initial = float(sdur_init)
    result.search_bounds = (float(lower), float(upper))

    logger.info(
        "Local timing refinement: init=%.6f ms, candidate=%.6f ms, "
        "final=%.6f ms, dtime=%.6f ms, rel_improvement=%.3f%%, "
        "edge=%s — %s",
        sdur_init * 1e3, candidate * 1e3, sdur * 1e3, dtime * 1e3,
        100.0 * relative_improvement, hits_edge, decision,
    )

    return sdur, dtime, result
