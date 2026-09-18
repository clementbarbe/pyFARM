"""EMG envelope, HRF convolution, regressor construction,
and envelope baseline correction.

Pipeline:
    Bandpass → Hilbert envelope → Normalize [0,1]
    → Baseline correction → ↓ 1000 Hz
    → HRF convolution → center at quiet baseline → derivatives → ↓ TR

For the "log" path, a compressive log transform is applied to the
envelope BEFORE convolution with the HRF.  This boosts low-amplitude
EMG activity relative to large bursts, creating a genuinely different
convolution profile from the linear path.
"""

import logging
import os
from math import gcd
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter as _percentile_filter
from scipy.signal import (
    fftconvolve, hilbert as _hilbert,
    butter, sosfiltfilt, resample_poly,
)
from scipy.stats import gamma as _gamma_dist
from scipy.io import savemat

from farm.preprocessing.filters import apply_bandpass

logger = logging.getLogger("farm.export.regressors")


# ═══════════════════════════════════════════════════════════════
#  Normalisation & transform primitives
# ═══════════════════════════════════════════════════════════════

def _normalize_range(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx - mn < 1e-30 else (x - mn) / (mx - mn)


def _log_compress(x: np.ndarray, gain: float = 50.0) -> np.ndarray:
    """Compressive logarithmic transform.

    Maps [0, 1] → [0, 1] with low values boosted::

        f(x) = log(1 + gain·x) / log(1 + gain)

    Effect with gain=50:
        x=0.01 → 0.10   (10× boost)
        x=0.10 → 0.46   (4.6× boost)
        x=0.50 → 0.83   (1.7× boost)
        x=1.00 → 1.00

    If gain ≤ 0, falls back to legacy ``log(x + 1) / log(2)``
    which is nearly linear on [0, 1].

    Parameters
    ----------
    x : 1-D array, values in [0, 1].
    gain : float — compression strength.  50 = moderate, 200 = strong.

    Returns
    -------
    1-D float64 array, values in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    if gain <= 0:
        # Legacy behaviour (nearly linear on [0, 1])
        return np.log(x + 1.0) / np.log(2.0)
    return np.log(1.0 + gain * x) / np.log(1.0 + gain)


def _center_at_quiet(
    convolved: np.ndarray,
    envelope: np.ndarray,
    quiet_threshold: float = 0.01,
) -> np.ndarray:
    """Center a convolved signal so that baseline = 0, peak = 1.

    The baseline is the median convolution value during quiet periods
    (where the source envelope is near zero).

    Undershoot is preserved as negative values.
    """
    convolved = np.asarray(convolved, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)

    quiet_mask = envelope < quiet_threshold
    n_quiet = int(np.sum(quiet_mask))

    if n_quiet >= 10:
        baseline = float(np.median(convolved[quiet_mask]))
    else:
        baseline = float(np.percentile(convolved, 5))
        logger.warning(
            "Only %d quiet samples — using 5th percentile fallback.", n_quiet)

    centered = convolved - baseline
    peak = float(np.max(centered))
    if peak < 1e-30:
        return np.zeros_like(centered)
    return centered / peak


def _scale_by_absmax(x: np.ndarray) -> np.ndarray:
    """Scale by max absolute value, preserving sign.  Range: [-1, +1]."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-30:
        return np.zeros_like(x)
    return x / peak


# ═══════════════════════════════════════════════════════════════
#  HRF
# ═══════════════════════════════════════════════════════════════

def spm_hrf(dt: float) -> np.ndarray:
    """SPM canonical double-gamma HRF at resolution *dt* (s)."""
    p = [6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 32.0]
    t = np.arange(0, p[6] + dt, dt)
    hrf = (
        _gamma_dist.pdf(t, p[0] / p[2], scale=p[2])
        - _gamma_dist.pdf(t, p[1] / p[3], scale=p[3]) / p[4]
    )
    return hrf / np.max(np.abs(hrf))


# ═══════════════════════════════════════════════════════════════
#  Envelope
# ═══════════════════════════════════════════════════════════════

def emg_envelope(
    ts_1d: np.ndarray, fsample: float, filter_order: int = 8,
) -> np.ndarray:
    """Hilbert envelope → normalise [0,1] → LP 10 Hz."""
    env = np.abs(_hilbert(ts_1d.astype(np.float64)))
    env = _normalize_range(env)
    if 10.0 < fsample / 2.0:
        sos = butter(filter_order, 10.0, btype="low", fs=fsample, output="sos")
        env = sosfiltfilt(sos, env)
    return env


# ═══════════════════════════════════════════════════════════════
#  Baseline correction
# ═══════════════════════════════════════════════════════════════

def _rolling_baseline(
    envelope: np.ndarray, srate: float,
    window_sec: float, percentile: float,
) -> np.ndarray:
    ds = max(1, int(round(srate / 50)))
    env_ds = envelope[::ds].astype(np.float64)
    srate_ds = srate / ds
    win = int(round(window_sec * srate_ds))
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    bl_ds = _percentile_filter(env_ds, percentile, size=win, mode="reflect")
    x_ds = np.arange(len(bl_ds), dtype=np.float64) * ds
    baseline = np.interp(np.arange(len(envelope), dtype=np.float64), x_ds, bl_ds)
    return baseline[: len(envelope)]


def _estimate_noise_std(corrected: np.ndarray) -> float:
    q25 = float(np.percentile(corrected, 25))
    noise_samples = corrected[corrected <= q25]
    if len(noise_samples) < 10:
        return float(np.std(corrected)) * 0.1
    med = float(np.median(noise_samples))
    mad = float(np.median(np.abs(noise_samples - med)))
    return mad / 0.6745 if mad > 0 else float(np.std(noise_samples))


def correct_envelope_baseline(
    envelope: np.ndarray, srate: float,
    method: str = "robust", percentile: float = 10.0,
    window_sec: float = 30.0, threshold_factor: float = 2.5,
) -> tuple:
    envelope = np.asarray(envelope, dtype=np.float64).ravel()
    info = {"method": method}

    if method == "none":
        return envelope.copy(), info

    baseline = _rolling_baseline(envelope, srate, window_sec, percentile)
    subtracted = np.maximum(0.0, envelope - baseline)
    info["baseline"] = baseline
    info["before_threshold"] = subtracted.copy()

    if method == "percentile":
        info["threshold_value"] = 0.0
        info["noise_std"] = 0.0
        return _normalize_range(subtracted), info

    noise_std = _estimate_noise_std(subtracted)
    threshold_value = threshold_factor * noise_std
    thresholded = subtracted.copy()
    thresholded[thresholded < threshold_value] = 0.0
    info["noise_std"] = noise_std
    info["threshold_value"] = threshold_value

    logger.info(
        "Envelope baseline: noise_std=%.4e, threshold=%.4e, "
        "%.1f%% zeroed",
        noise_std, threshold_value, 100.0 * np.mean(thresholded == 0),
    )
    return _normalize_range(thresholded), info


# ═══════════════════════════════════════════════════════════════
#  Regressor construction
# ═══════════════════════════════════════════════════════════════

def make_regressor(
    envelope: np.ndarray,
    fsample: float,
    n_volumes: int,
    tr: float,
    center_hrf: bool = True,
    log_compress_gain: float = 50.0,
    volume_onsets_sec: np.ndarray | None = None,
) -> dict:
    """Build regressors from a normalised EMG envelope.

    Parameters
    ----------
    envelope
        One-dimensional corrected envelope in [0, 1].
    fsample
        Sampling rate of envelope.
    n_volumes
        Number of fMRI volumes.
    tr
        Nominal TR in seconds. Used when real onset times are not supplied.
    center_hrf
        Center HRF-convolved regressors around quiet baseline.
    log_compress_gain
        Compression gain for log envelope pathway.
    volume_onsets_sec
        Optional real scanner volume onset times, relative to the beginning
        of the cropped EMG data. If supplied, these times are used exactly
        for TR-level sampling.

    Returns
    -------
    dict
        High-resolution and volume-level regressors.
    """
    ts = np.asarray(envelope, dtype=np.float64).ravel()

    if len(ts) < 2:
        raise ValueError("Envelope must contain at least two samples")

    if fsample <= 0:
        raise ValueError("fsample must be positive")

    if n_volumes < 1:
        raise ValueError("n_volumes must be >= 1")

    hrf = spm_hrf(1.0 / fsample)
    log_ts = _log_compress(ts, gain=log_compress_gain)

    if np.std(ts) > 1e-20 and np.std(log_ts) > 1e-20:
        corr = float(np.corrcoef(ts, log_ts)[0, 1])
    else:
        corr = 1.0

    logger.info(
        "Log compression gain=%.0f: corr(envelope, log_envelope)=%.4f",
        log_compress_gain,
        corr,
    )

    raw_conv = fftconvolve(ts, hrf, mode="full")[:len(ts)]
    raw_log_conv = fftconvolve(log_ts, hrf, mode="full")[:len(ts)]

    if center_hrf:
        conv = _center_at_quiet(raw_conv, ts)
        log_conv = _center_at_quiet(raw_log_conv, ts)

        dconv = _scale_by_absmax(np.gradient(conv, 1.0 / fsample))
        dlog_conv = _scale_by_absmax(np.gradient(log_conv, 1.0 / fsample))
    else:
        conv = _normalize_range(raw_conv)
        log_conv = _normalize_range(raw_log_conv)

        dconv = _normalize_range(np.gradient(conv, 1.0 / fsample))
        dlog_conv = _normalize_range(np.gradient(log_conv, 1.0 / fsample))

    mod_s = _normalize_range(ts)
    log_mod_full = _normalize_range(log_ts)

    dmod_full = _normalize_range(np.gradient(mod_s, 1.0 / fsample))
    dlog_mod_full = _normalize_range(
        np.gradient(log_mod_full, 1.0 / fsample)
    )

    time_conv = np.arange(len(conv), dtype=np.float64) / fsample

    if volume_onsets_sec is None:
        time_reg = np.arange(n_volumes, dtype=np.float64) * float(tr)
    else:
        time_reg = np.asarray(volume_onsets_sec, dtype=np.float64).ravel()

        if len(time_reg) != n_volumes:
            raise ValueError(
                "volume_onsets_sec must have n_volumes entries: "
                f"got {len(time_reg)}, expected {n_volumes}."
            )

    # np.interp avoids a rounding bias and uses the real temporal positions.
    reg = np.interp(time_reg, time_conv, conv, left=conv[0], right=conv[-1])
    dreg = np.interp(time_reg, time_conv, dconv, left=dconv[0], right=dconv[-1])

    log_reg = np.interp(
        time_reg,
        time_conv,
        log_conv,
        left=log_conv[0],
        right=log_conv[-1],
    )
    dlog_reg = np.interp(
        time_reg,
        time_conv,
        dlog_conv,
        left=dlog_conv[0],
        right=dlog_conv[-1],
    )

    mod = np.interp(time_reg, time_conv, mod_s, left=mod_s[0], right=mod_s[-1])
    log_mod = np.interp(
        time_reg,
        time_conv,
        log_mod_full,
        left=log_mod_full[0],
        right=log_mod_full[-1],
    )
    dmod = np.interp(
        time_reg,
        time_conv,
        dmod_full,
        left=dmod_full[0],
        right=dmod_full[-1],
    )
    dlog_mod = np.interp(
        time_reg,
        time_conv,
        dlog_mod_full,
        left=dlog_mod_full[0],
        right=dlog_mod_full[-1],
    )

    return dict(
        conv=conv,
        dconv=dconv,
        log_conv=log_conv,
        dlog_conv=dlog_conv,
        time_conv=time_conv,

        reg=reg,
        dreg=dreg,
        log_reg=log_reg,
        dlog_reg=dlog_reg,
        time_reg=time_reg,

        mod=mod,
        log_mod=log_mod,
        dmod=dmod,
        dlog_mod=dlog_mod,
    )

# ═══════════════════════════════════════════════════════════════
#  Per-channel .mat export
# ═══════════════════════════════════════════════════════════════

def _save_channel_mat(
    reg_dir: Path, ch_name: str, reginfo: dict,
    envelope_raw: np.ndarray, envelope_corrected: np.ndarray,
) -> str:
    mat_dict = {}
    for key, val in reginfo.items():
        mat_dict[key] = np.atleast_1d(val)
    mat_dict["envelope_raw"] = np.atleast_1d(envelope_raw)
    mat_dict["envelope_corrected"] = np.atleast_1d(envelope_corrected)
    path = str(reg_dir / f"{ch_name}_regressors.mat")
    savemat(path, mat_dict, do_compression=True)
    logger.debug("Per-channel MAT: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════
#  High-level export
# ═══════════════════════════════════════════════════════════════

def build_and_export_regressors(
    data_clean_crop: np.ndarray,
    srate: float,
    ch_names: list,
    n_vol: int,
    tr: float,
    bandpass: tuple,
    output_dir: str,
    basename: str,
    fig_dir: Path | None = None,
    new_fsample: int = 1000,
    envelope_baseline: str = "robust",
    envelope_percentile: float = 10.0,
    envelope_window_sec: float = 30.0,
    envelope_threshold_factor: float = 2.5,
    center_hrf: bool = True,
    log_compress_gain: float = 50.0,
    volume_onsets_sec: np.ndarray | None = None,
) -> dict:
    """Build EMG regressors for all channels and write to disk."""
    out = Path(output_dir)
    reg_dir = out / "regressors"
    reg_dir.mkdir(parents=True, exist_ok=True)

    all_regressors = {}
    srate_int = int(round(srate))

    for ch_i, ch_name in enumerate(ch_names):
        logger.info("Building regressors for %s …", ch_name)

        ts_bp = apply_bandpass(
            data_clean_crop[ch_i], srate, bandpass).astype(np.float64)
        envelope_raw = emg_envelope(ts_bp, srate)
        envelope_norm = _normalize_range(envelope_raw)

        envelope_corrected, correction_info = correct_envelope_baseline(
            envelope_norm, srate,
            method=envelope_baseline, percentile=envelope_percentile,
            window_sec=envelope_window_sec,
            threshold_factor=envelope_threshold_factor,
        )

        if fig_dir is not None:
            from farm.visualization.regressors import plot_envelope_correction
            plot_envelope_correction(
                envelope_norm, envelope_corrected, correction_info,
                srate, ch_name, fig_dir)

        if srate_int > new_fsample:
            g = gcd(new_fsample, srate_int)
            up_f, down_f = new_fsample // g, srate_int // g
            envelope_ds = resample_poly(
                envelope_corrected, up_f, down_f).astype(np.float64)
            fs_reg = float(new_fsample)
        else:
            envelope_ds = envelope_corrected.copy()
            fs_reg = float(srate)

        reginfo = make_regressor(
            envelope_ds,
            fs_reg,
            n_vol,
            tr,
            center_hrf=center_hrf,
            log_compress_gain=log_compress_gain,
            volume_onsets_sec=volume_onsets_sec,
        )
        all_regressors[ch_name] = reginfo

        mat_path = _save_channel_mat(
            reg_dir, ch_name, reginfo,
            envelope_norm, envelope_corrected)
        logger.info("Regressors for %s → %s", ch_name, mat_path)

        if fig_dir is not None:
            from farm.visualization.regressors import (
                plot_regressor, plot_regressor_panels)
            plot_regressor(
                reginfo, ch_name, fig_dir,
                envelope=envelope_corrected, srate_envelope=srate,
                centered=center_hrf)
            plot_regressor_panels(
                reginfo, ch_name, fig_dir,
                envelope=envelope_corrected, srate_envelope=srate,
                centered=center_hrf)

    mat_all = {}
    for ch_name_r, reginfo in all_regressors.items():
        pfx = ch_name_r.replace(" ", "_").replace("-", "_")
        for key, val in reginfo.items():
            mat_all[f"{pfx}_{key}"] = np.atleast_1d(val)
    mat_all_path = str(out / f"{basename}_regressors.mat")
    savemat(mat_all_path, mat_all, do_compression=True)
    logger.info("Consolidated MAT: %s (%.1f MB)",
                mat_all_path, os.path.getsize(mat_all_path) / 1e6)

    return all_regressors