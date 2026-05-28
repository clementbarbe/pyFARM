"""PCA residual cleanup (FARM step vi)."""

import logging

import numpy as np
from farm.preprocessing.filters import hpf_butter_1d
from farm.alignment.phase_shift import extract_plain_segments

logger = logging.getLogger("farm.pca.cleanup")


def pca_cleanup(
    vol_clean_signal: np.ndarray,
    vol_noise_signal: np.ndarray,
    onsets_up: np.ndarray,
    seg_len: int,
    valid_idx: np.ndarray,
    srate_up: float,
    scan_start_up: int,
    scan_stop_up: int,
    dtime_samp_up: int,
    time_section: float = 60.0,
    var_threshold: float = 5.0,
) -> np.ndarray:
    """Remove residual artifact structure via PCA on HPF'd residuals.

    Parameters
    ----------
    vol_clean_signal : 1-D array — phase-aligned + zero-filled signal.
    vol_noise_signal : 1-D array — template artifact + zero-filled.
    onsets_up : int64 array of slice onsets.
    seg_len : int — segment length (upsampled).
    valid_idx : int64 array — slice indices that have valid segments.
    srate_up : float — upsampled sampling rate.
    scan_start_up, scan_stop_up : int — scan region boundaries.
    dtime_samp_up : int — dead-time (upsampled samples).
    time_section : float — PCA section duration (s).
    var_threshold : float — minimum explained variance (%) per component.

    Returns
    -------
    clean_signal : 1-D float32 array, same length as input.
    """
    subtracted = (vol_clean_signal - vol_noise_signal).astype(np.float32)
    sub_hpf70 = hpf_butter_1d(subtracted, srate_up, 70.0)

    sub_segs_hpf, valid_pca = extract_plain_segments(
        sub_hpf70, onsets_up, seg_len, indices=valid_idx)
    sub_segs_raw, _ = extract_plain_segments(
        subtracted, onsets_up, seg_len, indices=valid_idx)
    noise_segs, _ = extract_plain_segments(
        vol_noise_signal, onsets_up, seg_len, indices=valid_idx)

    clean_signal = vol_clean_signal.copy()
    noise_signal = vol_noise_signal.copy()
    clean_signal[scan_start_up:scan_stop_up] = subtracted[scan_start_up:scan_stop_up]

    scan_duration = (
        onsets_up[-1] + 2 * seg_len + dtime_samp_up - onsets_up[0]
    ) / srate_up
    n_sections = max(1, int(round(scan_duration / time_section)))
    rows = np.arange(len(valid_pca), dtype=np.int64)
    rows_per_section = len(rows) / n_sections
    total_components = 0

    for sec in range(n_sections):
        start_row = int(round(sec * rows_per_section))
        stop_row = int(round((sec + 1) * rows_per_section))
        sec_rows = rows[start_row:stop_row]
        if len(sec_rows) < 2:
            continue

        M = sub_segs_hpf[sec_rows].T.astype(np.float64)
        mean_art = M.mean(axis=1)
        M_centered = M - M.mean(axis=0, keepdims=True)

        U, S, _ = np.linalg.svd(M_centered, full_matrices=False)
        eig = S ** 2
        var_pct = 100.0 * eig / (eig.sum() + 1e-30)
        n_comp = int(np.sum(var_pct > var_threshold))
        total_components += n_comp

        fitted = np.zeros_like(M_centered)
        if n_comp > 0:
            PC = U[:, :n_comp].copy()
            ptp = PC.max(axis=0) - PC.min(axis=0)
            ptp[ptp < 1e-12] = 1.0
            PC = PC / ptp * ptp[0]
            for j in range(M_centered.shape[1]):
                coeff, _, _, _ = np.linalg.lstsq(PC, M_centered[:, j], rcond=None)
                fitted[:, j] = PC @ coeff

        for j, row_idx in enumerate(sec_rows):
            slice_idx = int(valid_pca[row_idx])
            start = int(onsets_up[slice_idx])
            stop = start + seg_len
            if 0 <= start and stop <= len(clean_signal):
                clean_signal[start:stop] = (
                    sub_segs_raw[row_idx].astype(np.float64)
                    - fitted[:, j]
                    - mean_art
                ).astype(np.float32)
                noise_signal[start:stop] = (
                    noise_segs[row_idx].astype(np.float64)
                    + fitted[:, j]
                    + mean_art
                ).astype(np.float32)

    logger.info(
        "PCA: %d sections, %d total components retained (threshold %.1f%%)",
        n_sections, total_components, var_threshold,
    )
    return clean_signal.astype(np.float32)