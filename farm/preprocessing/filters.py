"""Frequency-domain filters used throughout the pipeline."""

import numpy as np
from scipy import signal as sig


def hpf_fir(x: np.ndarray, srate: float, cutoff: float) -> np.ndarray:
    """Zero-phase FIR high-pass filter (least-squares design).

    Parameters
    ----------
    x : ndarray, shape ``(n_channels, n_samples)``.
    srate : float — sampling rate in Hz.
    cutoff : float — cutoff frequency in Hz.

    Returns
    -------
    Filtered array, same shape and dtype as *x*.
    """
    numtaps = int(3.0 * srate / cutoff) | 1
    numtaps = max(numtaps, 101)
    h = sig.firls(
        numtaps,
        [0, cutoff * 0.5, cutoff, srate / 2],
        [0, 0, 1, 1],
        fs=srate,
    )
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        padlen = min(3 * numtaps, x.shape[1] - 1)
        out[ch] = sig.filtfilt(h, 1.0, x[ch], padlen=padlen).astype(x.dtype)
    return out


def hpf_butter_1d(x: np.ndarray, srate: float, cutoff: float) -> np.ndarray:
    """1-D Butterworth high-pass (order 4, zero-phase)."""
    sos = sig.butter(4, cutoff, btype="high", fs=srate, output="sos")
    return sig.sosfiltfilt(sos, x).astype(x.dtype)


def lpf_butter(x: np.ndarray, srate: float, cutoff: float) -> np.ndarray:
    """Multi-channel Butterworth low-pass (order 4, zero-phase).

    Parameters
    ----------
    x : ndarray, shape ``(n_channels, n_samples)``.
    """
    sos = sig.butter(4, cutoff, btype="low", fs=srate, output="sos")
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        out[ch] = sig.sosfiltfilt(sos, x[ch]).astype(x.dtype)
    return out


def apply_bandpass(data_1d: np.ndarray, srate: float,
                   bandpass: tuple) -> np.ndarray:
    """Convenience band-pass (Butterworth order 4, zero-phase).

    Parameters
    ----------
    data_1d : 1-D signal.
    bandpass : ``(low_hz, high_hz)`` or *None* to skip.

    Returns
    -------
    Filtered signal as float32.
    """
    ts = data_1d.astype(np.float64)
    if bandpass:
        sos_hp = sig.butter(4, bandpass[0], "high", fs=srate, output="sos")
        sos_lp = sig.butter(4, bandpass[1], "low", fs=srate, output="sos")
        ts = sig.sosfiltfilt(sos_hp, ts)
        ts = sig.sosfiltfilt(sos_lp, ts)
    return ts.astype(np.float32)