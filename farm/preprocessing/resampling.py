"""Up- and down-sampling helpers."""

import numpy as np
from scipy import signal as sig


def upsample(
    data: np.ndarray,
    srate: float,
    vol_onsets: np.ndarray,
    factor: int,
) -> tuple:
    """Polyphase upsampling of all channels.

    Parameters
    ----------
    data : ndarray, shape ``(n_ch, n_samples)``.
    srate : float — original sampling rate.
    vol_onsets : 1-D int array — volume onsets in *original* samples.
    factor : int — upsampling factor.

    Returns
    -------
    data_up : ndarray, shape ``(n_ch, n_samples * factor)`` (float32).
    srate_up : float.
    vol_onsets_up : int64 array.
    """
    n_ch, n_samp = data.shape
    srate_up = srate * factor
    n_up = n_samp * factor
    data_up = np.empty((n_ch, n_up), dtype=np.float32)
    for ch in range(n_ch):
        r = sig.resample_poly(data[ch].astype(np.float64), factor, 1)
        data_up[ch] = r[:n_up].astype(np.float32)
    vol_onsets_up = (vol_onsets * factor).astype(np.int64)
    return data_up, srate_up, vol_onsets_up


def downsample(
    data_up: np.ndarray,
    srate_up: float,
    factor: int,
    target_length: int,
) -> np.ndarray:
    """Polyphase downsampling of all channels.

    Parameters
    ----------
    data_up : ndarray, shape ``(n_ch, n_samples_up)``.
    srate_up : float — current (upsampled) sampling rate.
    factor : int — downsampling factor.
    target_length : int — expected output length.

    Returns
    -------
    ndarray, shape ``(n_ch, target_length)``, float32.
    """
    n_ch = data_up.shape[0]
    out = np.empty((n_ch, target_length), dtype=np.float32)
    for ch in range(n_ch):
        r = sig.resample_poly(data_up[ch].astype(np.float64), 1, factor)
        out[ch] = r[:target_length].astype(np.float32)
    return out