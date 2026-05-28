"""EMG envelope, HRF convolution, and regressor construction.

Reproduces the logic of ``farm_emg_regressor`` / ``farm_make_regressor``:

    Bandpass → Hilbert envelope → Normalize [0,1]
    → ↓ 1000 Hz → HRF convolution → derivatives + log → ↓ TR
"""

import logging
import os
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, hilbert as _hilbert, butter, sosfiltfilt, resample_poly
from scipy.stats import gamma as _gamma_dist
from scipy.io import savemat

from farm.preprocessing.filters import apply_bandpass

logger = logging.getLogger("farm.export.regressors")


# ── Primitives ──────────────────────────────────────────────────

def _normalize_range(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx - mn < 1e-30 else (x - mn) / (mx - mn)


def _log_transform(x: np.ndarray) -> np.ndarray:
    """``log(x - min(x) + 1)``."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(x - x.min() + 1.0)


def spm_hrf(dt: float) -> np.ndarray:
    """SPM canonical double-gamma HRF at resolution *dt* (s)."""
    p = [6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 32.0]
    t = np.arange(0, p[6] + dt, dt)
    hrf = (
        _gamma_dist.pdf(t, p[0] / p[2], scale=p[2])
        - _gamma_dist.pdf(t, p[1] / p[3], scale=p[3]) / p[4]
    )
    return hrf / np.max(np.abs(hrf))


def emg_envelope(
    ts_1d: np.ndarray, fsample: float, filter_order: int = 8
) -> np.ndarray:
    """Hilbert envelope → normalise [0,1] → LP 10 Hz."""
    env = np.abs(_hilbert(ts_1d.astype(np.float64)))
    env = _normalize_range(env)
    if 10.0 < fsample / 2.0:
        sos = butter(filter_order, 10.0, btype="low", fs=fsample, output="sos")
        env = sosfiltfilt(sos, env)
    return env


def make_regressor(
    envelope: np.ndarray, fsample: float, n_volumes: int, tr: float
) -> dict:
    """Build a full set of regressors from a normalised envelope.

    Returns a dict with keys: ``conv``, ``dconv``, ``log_conv``,
    ``dlog_conv``, ``time_conv``, ``reg``, ``dreg``, ``log_reg``,
    ``dlog_reg``, ``time_reg``, ``mod``, ``log_mod``, ``dmod``,
    ``dlog_mod``.
    """
    ts = np.asarray(envelope, dtype=np.float64).ravel()
    hrf = spm_hrf(1.0 / fsample)

    conv = _normalize_range(fftconvolve(ts, hrf, mode="full")[: len(ts)])
    log_ts = _normalize_range(_log_transform(ts))
    log_conv = _normalize_range(fftconvolve(log_ts, hrf, mode="full")[: len(ts)])

    dconv = _normalize_range(np.concatenate([[0], np.diff(conv)]))
    dlog_conv = _normalize_range(np.concatenate([[0], np.diff(log_conv)]))

    mod_s = _normalize_range(ts)
    log_mod = _normalize_range(log_ts)
    dmod = _normalize_range(np.concatenate([[0], np.diff(mod_s)]))
    dlog_mod = _normalize_range(np.concatenate([[0], np.diff(log_mod)]))

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


# ── High-level export ──────────────────────────────────────────

def build_and_export_regressors(
    data_clean_crop: np.ndarray,
    srate: float,
    ch_names: list,
    n_vol: int,
    tr: float,
    bandpass: tuple,
    output_dir: str,
    basename: str,
    new_fsample: int = 1000,
) -> dict:
    """Build EMG regressors for all channels and write to disk.

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
        ts_bp = apply_bandpass(data_clean_crop[ch_i], srate, bandpass).astype(np.float64)
        envelope = emg_envelope(ts_bp, srate)
        envelope_norm = _normalize_range(envelope)

        # Downsample to new_fsample
        if srate_int > new_fsample:
            g = gcd(new_fsample, srate_int)
            up_f, down_f = new_fsample // g, srate_int // g
            envelope_ds = resample_poly(envelope_norm, up_f, down_f).astype(np.float64)
            fs_reg = float(new_fsample)
        else:
            envelope_ds = envelope_norm.copy()
            fs_reg = float(srate)

        reginfo = make_regressor(envelope_ds, fs_reg, n_vol, tr)
        all_regressors[ch_name] = reginfo

        # Save per-channel files
        for rk in ["reg", "dreg", "log_reg", "dlog_reg",
                    "mod", "log_mod", "dmod", "dlog_mod"]:
            np.save(str(reg_dir / f"{ch_name}_{rk}.npy"), reginfo[rk])
        np.save(str(reg_dir / f"{ch_name}_time_reg.npy"), reginfo["time_reg"])
        np.save(str(reg_dir / f"{ch_name}_conv_hires.npy"), reginfo["conv"])
        np.save(str(reg_dir / f"{ch_name}_time_conv.npy"), reginfo["time_conv"])
        np.save(str(reg_dir / f"{ch_name}_envelope.npy"), envelope_norm)

        logger.info("Regressors for %s saved.", ch_name)

    # Consolidated MAT
    mat_reg = {}
    for ch_name, reginfo in all_regressors.items():
        pfx = ch_name.replace(" ", "_").replace("-", "_")
        for key, val in reginfo.items():
            mat_reg[f"{pfx}_{key}"] = val
    mat_path = str(out / f"{basename}_regressors.mat")
    savemat(mat_path, mat_reg, do_compression=True)
    logger.info("Consolidated MAT: %s (%.1f MB)", mat_path, os.path.getsize(mat_path) / 1e6)

    return all_regressors