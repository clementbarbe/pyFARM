"""EMG envelope, HRF convolution, regressor construction,
and envelope baseline correction.

Reproduces the logic of ``farm_emg_regressor`` / ``farm_make_regressor``:

    Bandpass → Hilbert envelope → Normalize [0,1]
    → Baseline correction → ↓ 1000 Hz
    → HRF convolution → **center at baseline** → derivatives + log → ↓ TR

When ``center_hrf=True``, the HRF-convolved signals are centered so that
quiet periods sit at zero.  The HRF undershoot is preserved as negative
values.  Derivatives are computed on the centered signal.
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
#  Normalisation primitives
# ═══════════════════════════════════════════════════════════════

def _normalize_range(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]  (legacy behaviour)."""
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx - mn < 1e-30 else (x - mn) / (mx - mn)


def _center_and_scale(
    x: np.ndarray, baseline_percentile: float = 10.0,
) -> np.ndarray:
    """Center so that baseline = 0, scale so that peak = 1.

    Negative values (HRF undershoot) are **preserved**.

    The baseline is estimated as a low percentile of the signal,
    which is robust even if the recording starts with an active
    period — the percentile will find quiet periods elsewhere.

    Parameters
    ----------
    x : 1-D array — raw convolved signal.
    baseline_percentile : float — percentile used to estimate the
        resting level (default 10).

    Returns
    -------
    1-D float64 array.  Baseline ≈ 0, peak ≈ 1, undershoot < 0.
    """
    x = np.asarray(x, dtype=np.float64)
    baseline = float(np.percentile(x, baseline_percentile))
    centered = x - baseline
    peak = float(np.max(centered))
    if peak < 1e-30:
        return np.zeros_like(centered)
    return centered / peak


def _scale_by_absmax(x: np.ndarray) -> np.ndarray:
    """Scale by max absolute value, preserving sign and zero baseline.

    Used for derivatives of centered signals: the derivative of a
    signal that is 0 at baseline is naturally 0 at baseline, so we
    only need to scale the amplitude.

    Returns
    -------
    1-D float64 array with values in [-1, +1].
    """
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-30:
        return np.zeros_like(x)
    return x / peak


def _log_transform(x: np.ndarray) -> np.ndarray:
    """``log(x - min(x) + 1)``."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(x - x.min() + 1.0)


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
    envelope: np.ndarray,
    srate: float,
    window_sec: float,
    percentile: float,
) -> np.ndarray:
    """Estimate a slowly-varying noise floor via rolling percentile."""
    ds = max(1, int(round(srate / 50)))
    env_ds = envelope[::ds].astype(np.float64)
    srate_ds = srate / ds

    win = int(round(window_sec * srate_ds))
    win = max(3, win)
    if win % 2 == 0:
        win += 1

    bl_ds = _percentile_filter(env_ds, percentile, size=win, mode="reflect")

    x_ds = np.arange(len(bl_ds), dtype=np.float64) * ds
    baseline = np.interp(
        np.arange(len(envelope), dtype=np.float64), x_ds, bl_ds,
    )
    return baseline[: len(envelope)]


def _estimate_noise_std(corrected: np.ndarray) -> float:
    """Robustly estimate the standard deviation of the noise floor."""
    q25 = float(np.percentile(corrected, 25))
    noise_samples = corrected[corrected <= q25]
    if len(noise_samples) < 10:
        return float(np.std(corrected)) * 0.1
    med = float(np.median(noise_samples))
    mad = float(np.median(np.abs(noise_samples - med)))
    return mad / 0.6745 if mad > 0 else float(np.std(noise_samples))


def correct_envelope_baseline(
    envelope: np.ndarray,
    srate: float,
    method: str = "robust",
    percentile: float = 10.0,
    window_sec: float = 30.0,
    threshold_factor: float = 2.5,
) -> tuple:
    """Remove the residual noise floor from an EMG envelope.

    Parameters
    ----------
    envelope : 1-D array — normalised [0, 1] EMG envelope.
    srate : float — sampling rate of *envelope* (Hz).
    method : ``"none"`` | ``"percentile"`` | ``"robust"``
    percentile : float — percentile for baseline (default 10).
    window_sec : float — rolling window in seconds (default 30).
    threshold_factor : float — values below
        ``factor × noise_std`` are zeroed (default 2.5).

    Returns
    -------
    corrected : 1-D float64 array, normalised [0, 1].
    info : dict — diagnostic data for plotting.
    """
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
        "Envelope baseline correction: noise_std=%.4e, "
        "threshold=%.4e (%.1f × σ), "
        "%.1f%% of samples zeroed",
        noise_std, threshold_value, threshold_factor,
        100.0 * np.mean(thresholded == 0),
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
    center_hrf_percentile: float = 10.0,
) -> dict:
    """Build a full set of regressors from a normalised envelope.

    Parameters
    ----------
    envelope : 1-D array — normalised [0, 1] EMG envelope.
    fsample : float — sampling rate of *envelope*.
    n_volumes : int — number of fMRI volumes.
    tr : float — repetition time (s).
    center_hrf : bool — if *True*, HRF-convolved signals are centered
        so baseline = 0 and the HRF undershoot is preserved as
        negative values.  Derivatives are computed on this centered
        signal.  If *False*, legacy [0, 1] normalisation is used.
    center_hrf_percentile : float — percentile used to estimate the
        convolution baseline (default 10).

    Returns
    -------
    dict with keys: ``conv``, ``dconv``, ``log_conv``, ``dlog_conv``,
    ``time_conv``, ``reg``, ``dreg``, ``log_reg``, ``dlog_reg``,
    ``time_reg``, ``mod``, ``log_mod``, ``dmod``, ``dlog_mod``.

    When ``center_hrf=True``:
    - ``conv``, ``log_conv`` have baseline ≈ 0, peak ≈ 1, undershoot < 0.
    - ``dconv``, ``dlog_conv`` are in [-1, +1], baseline ≈ 0.
    - ``mod``, ``log_mod``, ``dmod``, ``dlog_mod`` remain in [0, 1].
    """
    ts = np.asarray(envelope, dtype=np.float64).ravel()
    hrf = spm_hrf(1.0 / fsample)

    # ── Raw convolutions ─────────────────────────────────────
    raw_conv = fftconvolve(ts, hrf, mode="full")[: len(ts)]
    log_ts = _log_transform(ts)
    raw_log_conv = fftconvolve(log_ts, hrf, mode="full")[: len(ts)]

    if center_hrf:
        # ── Centered mode: baseline = 0, undershoot preserved ──
        pct = center_hrf_percentile

        conv = _center_and_scale(raw_conv, pct)
        log_conv = _center_and_scale(raw_log_conv, pct)

        # Derivatives on centered signals, then scale
        dconv = _scale_by_absmax(np.concatenate([[0], np.diff(conv)]))
        dlog_conv = _scale_by_absmax(np.concatenate([[0], np.diff(log_conv)]))

        logger.debug(
            "HRF centering: conv range [%.3f, %.3f], "
            "log_conv range [%.3f, %.3f]",
            conv.min(), conv.max(), log_conv.min(), log_conv.max(),
        )
    else:
        # ── Legacy mode: everything in [0, 1] ───────────────
        conv = _normalize_range(raw_conv)
        log_conv = _normalize_range(raw_log_conv)
        dconv = _normalize_range(np.concatenate([[0], np.diff(conv)]))
        dlog_conv = _normalize_range(np.concatenate([[0], np.diff(log_conv)]))

    # ── Non-convolved modulations (always [0, 1]) ────────────
    mod_s = _normalize_range(ts)
    log_mod = _normalize_range(log_ts)
    dmod = _normalize_range(np.concatenate([[0], np.diff(mod_s)]))
    dlog_mod = _normalize_range(np.concatenate([[0], np.diff(log_mod)]))

    # ── Downsample to TR ─────────────────────────────────────
    time_conv = np.arange(len(conv)) / fsample
    idx = np.round(np.linspace(0, len(time_conv) - 1, n_volumes)).astype(int)
    time_reg = time_conv[idx]

    return dict(
        conv=conv, dconv=dconv, log_conv=log_conv, dlog_conv=dlog_conv,
        time_conv=time_conv,
        reg=conv[idx], dreg=dconv[idx],
        log_reg=log_conv[idx], dlog_reg=dlog_conv[idx],
        time_reg=time_reg,
        mod=mod_s[idx], log_mod=log_mod[idx],
        dmod=dmod[idx], dlog_mod=dlog_mod[idx],
    )


# ═══════════════════════════════════════════════════════════════
#  Per-channel .mat export
# ═══════════════════════════════════════════════════════════════

def _save_channel_mat(
    reg_dir: Path,
    ch_name: str,
    reginfo: dict,
    envelope_raw: np.ndarray,
    envelope_corrected: np.ndarray,
) -> str:
    """Save all regressors and envelopes for one channel as a single .mat."""
    mat_dict = {}
    for key, val in reginfo.items():
        mat_dict[key] = np.atleast_1d(val)
    mat_dict["envelope_raw"] = np.atleast_1d(envelope_raw)
    mat_dict["envelope_corrected"] = np.atleast_1d(envelope_corrected)

    path = str(reg_dir / f"{ch_name}_regressors.mat")
    savemat(path, mat_dict, do_compression=True)
    logger.debug("Per-channel MAT: %s (%.1f kB)", path,
                 os.path.getsize(path) / 1e3)
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
    center_hrf_percentile: float = 10.0,
) -> dict:
    """Build EMG regressors for all channels and write to disk.

    Parameters
    ----------
    data_clean_crop : ndarray, shape ``(n_ch, n_samples)``.
    srate : float — original sampling rate.
    ch_names : list of str.
    n_vol : int — number of volumes.
    tr : float — repetition time (s).
    bandpass : tuple — ``(low_hz, high_hz)``.
    output_dir : str — base output directory.
    basename : str — file prefix.
    fig_dir : Path or None — if set, diagnostic plots are saved here.
    new_fsample : int — target sampling rate for regressors (default 1000).
    envelope_baseline : str — correction method.
    envelope_percentile : float — percentile for rolling baseline.
    envelope_window_sec : float — window length (s) for rolling baseline.
    envelope_threshold_factor : float — noise threshold multiplier.
    center_hrf : bool — center HRF convolutions at baseline = 0.
    center_hrf_percentile : float — percentile for baseline estimation.

    Returns
    -------
    dict mapping channel name → regressor dict.
    """
    out = Path(output_dir)
    reg_dir = out / "regressors"
    reg_dir.mkdir(parents=True, exist_ok=True)

    all_regressors = {}
    srate_int = int(round(srate))

    for ch_i, ch_name in enumerate(ch_names):
        logger.info("Building regressors for %s …", ch_name)

        # ── 1. Bandpass ──────────────────────────────────────
        ts_bp = apply_bandpass(
            data_clean_crop[ch_i], srate, bandpass,
        ).astype(np.float64)

        # ── 2. Hilbert envelope ──────────────────────────────
        envelope_raw = emg_envelope(ts_bp, srate)
        envelope_norm = _normalize_range(envelope_raw)

        # ── 3. Baseline correction ───────────────────────────
        envelope_corrected, correction_info = correct_envelope_baseline(
            envelope_norm, srate,
            method=envelope_baseline,
            percentile=envelope_percentile,
            window_sec=envelope_window_sec,
            threshold_factor=envelope_threshold_factor,
        )

        # ── 4. Diagnostic plot: baseline correction ──────────
        if fig_dir is not None:
            from farm.visualization.regressors import (
                plot_envelope_correction,
            )
            plot_envelope_correction(
                envelope_norm, envelope_corrected, correction_info,
                srate, ch_name, fig_dir,
            )

        # ── 5. Downsample to new_fsample ─────────────────────
        if srate_int > new_fsample:
            g = gcd(new_fsample, srate_int)
            up_f, down_f = new_fsample // g, srate_int // g
            envelope_ds = resample_poly(
                envelope_corrected, up_f, down_f,
            ).astype(np.float64)
            fs_reg = float(new_fsample)
        else:
            envelope_ds = envelope_corrected.copy()
            fs_reg = float(srate)

        # ── 6. Build regressors ──────────────────────────────
        reginfo = make_regressor(
            envelope_ds, fs_reg, n_vol, tr,
            center_hrf=center_hrf,
            center_hrf_percentile=center_hrf_percentile,
        )
        all_regressors[ch_name] = reginfo

        if center_hrf:
            logger.info(
                "%s regressors: conv range [%.3f, %.3f], "
                "baseline ≈ 0, undershoot preserved",
                ch_name, reginfo["conv"].min(), reginfo["conv"].max(),
            )

        # ── 7. Save per-channel .mat ─────────────────────────
        mat_path = _save_channel_mat(
            reg_dir, ch_name, reginfo,
            envelope_norm, envelope_corrected,
        )
        logger.info("Regressors for %s → %s", ch_name, mat_path)

        # ── 8. Regressor overview plots ──────────────────────
        if fig_dir is not None:
            from farm.visualization.regressors import (
                plot_regressor, plot_regressor_panels,
            )
            plot_regressor(
                reginfo, ch_name, fig_dir,
                envelope=envelope_corrected, srate_envelope=srate,
                centered=center_hrf,
            )
            plot_regressor_panels(
                reginfo, ch_name, fig_dir,
                envelope=envelope_corrected, srate_envelope=srate,
                centered=center_hrf,
            )

    # ── Consolidated .mat (all channels) ─────────────────────
    mat_all = {}
    for ch_name_r, reginfo in all_regressors.items():
        pfx = ch_name_r.replace(" ", "_").replace("-", "_")
        for key, val in reginfo.items():
            mat_all[f"{pfx}_{key}"] = np.atleast_1d(val)
    mat_all_path = str(out / f"{basename}_regressors.mat")
    savemat(mat_all_path, mat_all, do_compression=True)
    logger.info(
        "Consolidated MAT: %s (%.1f MB)",
        mat_all_path, os.path.getsize(mat_all_path) / 1e6,
    )

    return all_regressors