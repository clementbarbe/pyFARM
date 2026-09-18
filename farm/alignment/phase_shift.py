"""FFT-based sub-sample phase shifting and segment I/O."""

import numpy as np


def fft_shift(segments: np.ndarray, shifts: np.ndarray | float) -> np.ndarray:
    """Apply sub-sample time shifts through FFT phase rotation.

    Convention
    ----------
    A positive ``shift`` delays the signal:

        y[n] = x[n - shift]

    Therefore, to extract a segment beginning at a rounded sample index while
    aligning it to an exact onset occurring later by ``round_error`` samples,
    call this function with ``shift=-round_error``.

    Parameters
    ----------
    segments
        1-D array ``(n_samples,)`` or 2-D array
        ``(n_segments, n_samples)``.
    shifts
        Scalar or one shift per segment, expressed in samples.

    Returns
    -------
    ndarray
        Shifted signal with same shape and dtype as input.
    """
    segments = np.asarray(segments)
    if segments.ndim not in (1, 2):
        raise ValueError(
            f"segments must be 1-D or 2-D, got shape {segments.shape}"
        )

    was_1d = segments.ndim == 1
    original_dtype = segments.dtype

    if was_1d:
        work = segments[np.newaxis, :].astype(np.float64, copy=False)
    else:
        work = segments.astype(np.float64, copy=False)

    n_seg, n_samp = work.shape
    if n_samp == 0:
        return segments.copy()

    shifts_arr = np.asarray(shifts, dtype=np.float64)

    if shifts_arr.ndim == 0:
        shifts_arr = np.full(n_seg, float(shifts_arr), dtype=np.float64)
    else:
        shifts_arr = shifts_arr.ravel()
        if len(shifts_arr) != n_seg:
            raise ValueError(
                f"Expected one shift per segment ({n_seg}), "
                f"received {len(shifts_arr)}."
            )

    spectrum = np.fft.rfft(work, axis=1)
    frequencies = np.arange(spectrum.shape[1], dtype=np.float64)

    # Positive shift = delay: y[n] = x[n - shift].
    phase = -2.0 * np.pi * shifts_arr[:, None] * frequencies[None, :] / n_samp
    spectrum *= np.exp(1j * phase)

    shifted = np.fft.irfft(spectrum, n=n_samp, axis=1)
    out = shifted.astype(original_dtype, copy=False)

    return out[0] if was_1d else out


def extract_aligned_segments(
    signal_1d: np.ndarray,
    onsets: np.ndarray,
    seg_len: int,
    round_errors: np.ndarray,
    padding: int = 10,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract segments then align them to their exact fractional onsets.

    ``onsets`` contains rounded sample indices. ``round_errors`` is defined as:

        exact_onset - rounded_onset

    If exact onset is +0.25 samples after the rounded index, the signal must
    be advanced by 0.25 samples, hence ``fft_shift(..., -round_error)``.

    Parameters
    ----------
    signal_1d
        1-D signal.
    onsets
        Rounded onset indices.
    seg_len
        Segment length in samples.
    round_errors
        Exact onset minus rounded onset, in samples.
    padding
        Extra samples used to reduce circular FFT edge effects.
    indices
        Optional subset of global onset indices.

    Returns
    -------
    segments
        ``(n_valid, seg_len)`` float32 array.
    valid_idx
        Global indices corresponding to rows in ``segments``.
    """
    signal_1d = np.asarray(signal_1d)
    onsets = np.asarray(onsets, dtype=np.int64)
    round_errors = np.asarray(round_errors, dtype=np.float64)

    if seg_len < 1:
        raise ValueError(f"seg_len must be >= 1, got {seg_len}")

    if len(onsets) != len(round_errors):
        raise ValueError("onsets and round_errors must have identical lengths")

    if indices is None:
        indices = np.arange(len(onsets), dtype=np.int64)

    indices = np.asarray(indices, dtype=np.int64)
    half = int(padding // 2)

    segs = []
    valid_idx = []

    for idx in indices:
        if idx < 0 or idx >= len(onsets):
            continue

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

    segs_arr = np.asarray(segs, dtype=np.float32)
    valid_arr = np.asarray(valid_idx, dtype=np.int64)

    # Important: negative sign, see docstring.
    shifted = fft_shift(segs_arr, -round_errors[valid_arr])

    return shifted[:, half:half + seg_len].astype(np.float32), valid_arr


def extract_plain_segments(
    signal_1d: np.ndarray,
    onsets: np.ndarray,
    seg_len: int,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract segments without fractional phase correction."""
    signal_1d = np.asarray(signal_1d)
    onsets = np.asarray(onsets, dtype=np.int64)

    if seg_len < 1:
        raise ValueError(f"seg_len must be >= 1, got {seg_len}")

    if indices is None:
        indices = np.arange(len(onsets), dtype=np.int64)

    segs = []
    valid_idx = []

    for idx in np.asarray(indices, dtype=np.int64):
        if idx < 0 or idx >= len(onsets):
            continue

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
    """Write segment rows back into a copy of ``signal_1d``."""
    out = np.asarray(signal_1d).copy()
    onsets = np.asarray(onsets, dtype=np.int64)
    segments = np.asarray(segments)
    seg_indices = np.asarray(seg_indices, dtype=np.int64)

    if segments.ndim != 2:
        raise ValueError("segments must be a 2-D array")

    if len(segments) != len(seg_indices):
        raise ValueError("segments and seg_indices must have same row count")

    for row, idx in enumerate(seg_indices):
        if idx < 0 or idx >= len(onsets):
            continue

        start = int(onsets[idx])
        stop = start + segments.shape[1]

        if 0 <= start and stop <= len(out):
            out[start:stop] = segments[row]

    return out.astype(signal_1d.dtype, copy=False)