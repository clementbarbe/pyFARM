"""FFT-based sub-sample phase shifting and segment I/O."""

import numpy as np


def fft_shift(segments: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply sub-sample time shifts via FFT phase rotation.

    Parameters
    ----------
    segments : 1-D or 2-D array (n_segments, n_samples).
    shifts : scalar or 1-D array, one shift per segment (in samples).

    Returns
    -------
    Shifted array, same shape and dtype.
    """
    was_1d = segments.ndim == 1
    if was_1d:
        segments = segments[np.newaxis, :]
        shifts = np.atleast_1d(shifts)
    n_seg, n_samp = segments.shape
    spectrum = np.fft.rfft(segments.astype(np.float64), axis=1)
    k = np.arange(spectrum.shape[1], dtype=np.float64)
    phase = -2.0 * np.pi * shifts[:, None] * k[None, :] / n_samp
    spectrum *= np.exp(1j * phase)
    shifted = np.fft.irfft(spectrum, n=n_samp, axis=1)
    out = shifted.astype(segments.dtype)
    return out[0] if was_1d else out


def extract_aligned_segments(
    signal_1d: np.ndarray,
    onsets: np.ndarray,
    seg_len: int,
    round_errors: np.ndarray,
    padding: int = 10,
    indices: np.ndarray | None = None,
) -> tuple:
    """Extract segments from *signal_1d*, then FFT-shift each to
    compensate its rounding error.

    Parameters
    ----------
    signal_1d : 1-D signal array.
    onsets : int64 array — rounded onset samples.
    seg_len : int — segment length.
    round_errors : float64 array — sub-sample rounding errors.
    padding : int — extra samples on each side for the phase shift.
    indices : int array or None — subset of segment indices to extract.

    Returns
    -------
    segments : ndarray, shape ``(n_valid, seg_len)``, float32.
    valid_idx : int64 array — indices of successfully extracted segments.
    """
    if indices is None:
        indices = np.arange(len(onsets), dtype=np.int64)
    half = padding // 2
    segs, valid_idx = [], []
    for idx in np.asarray(indices, dtype=np.int64):
        start = int(onsets[idx]) - half
        stop = int(onsets[idx]) + seg_len + half
        if start < 0 or stop > len(signal_1d):
            continue
        segs.append(signal_1d[start:stop])
        valid_idx.append(int(idx))
    if not segs:
        return (
            np.empty((0, seg_len), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    segs = np.asarray(segs, dtype=np.float32)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    segs = fft_shift(segs, round_errors[valid_idx])
    return segs[:, half : half + seg_len], valid_idx


def extract_plain_segments(
    signal_1d: np.ndarray,
    onsets: np.ndarray,
    seg_len: int,
    indices: np.ndarray | None = None,
) -> tuple:
    """Extract segments *without* phase correction.

    Returns
    -------
    segments : ndarray, shape ``(n_valid, seg_len)``, float32.
    valid_idx : int64 array.
    """
    if indices is None:
        indices = np.arange(len(onsets), dtype=np.int64)
    segs, valid_idx = [], []
    for idx in np.asarray(indices, dtype=np.int64):
        start = int(onsets[idx])
        stop = start + seg_len
        if start < 0 or stop > len(signal_1d):
            continue
        segs.append(signal_1d[start:stop])
        valid_idx.append(int(idx))
    if not segs:
        return (
            np.empty((0, seg_len), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.asarray(segs, dtype=np.float32),
        np.asarray(valid_idx, dtype=np.int64),
    )


def overwrite_segments(
    signal_1d: np.ndarray,
    onsets: np.ndarray,
    segments: np.ndarray,
    seg_indices: np.ndarray,
) -> np.ndarray:
    """Write *segments* back into a copy of *signal_1d*.

    Parameters
    ----------
    signal_1d : 1-D array.
    onsets : int64 onset array (full set).
    segments : 2-D array, shape ``(n_segs, seg_len)``.
    seg_indices : int64 array mapping rows of *segments* to
        indices of *onsets*.

    Returns
    -------
    Copy of *signal_1d* with the specified segments overwritten.
    """
    out = signal_1d.copy()
    for row, idx in enumerate(np.asarray(seg_indices, dtype=np.int64)):
        start = int(onsets[idx])
        stop = start + segments.shape[1]
        if 0 <= start and stop <= len(out):
            out[start:stop] = segments[row]
    return out.astype(signal_1d.dtype, copy=False)