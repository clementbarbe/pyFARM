"""Conservative global residual PCA cleanup (FARM step vi).

One PCA is fitted per time section, not per acquisition group.  Components are
removed only when they explain enough residual variance *and* resemble the
artifact-template subspace.  This keeps the signal-preservation gate from the
quality-first branch while avoiding hundreds of independent group-wise fits
that can create slice-periodic discontinuities.
"""

from __future__ import annotations

import logging
import numpy as np

from farm.preprocessing.filters import hpf_butter_1d
from farm.alignment.phase_shift import extract_plain_segments

logger = logging.getLogger("farm.pca.cleanup")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-20:
        return 0.0
    return float(np.dot(a, b) / den)


def _artifact_basis(noise_matrix: np.ndarray, max_basis: int = 8) -> list[np.ndarray]:
    """Build a compact temporal basis from estimated artifact segments.

    Parameters
    ----------
    noise_matrix : samples x occurrences
        High-pass-filtered template-artifact segments from the same section.
    """
    N = np.asarray(noise_matrix, dtype=np.float64)
    if N.ndim != 2 or N.size == 0 or np.max(np.abs(N)) < 1e-20:
        return []

    basis: list[np.ndarray] = []
    mean_ref = N.mean(axis=1)
    if np.linalg.norm(mean_ref - mean_ref.mean()) > 1e-20:
        basis.append(mean_ref)

    N0 = N - N.mean(axis=0, keepdims=True)
    if np.max(np.abs(N0)) > 1e-20:
        U, S, _ = np.linalg.svd(N0, full_matrices=False)
        if len(S):
            energy = S * S
            pct = 100.0 * energy / (energy.sum() + 1e-30)
            keep = np.flatnonzero(pct > 1.0)[: int(max_basis)]
            basis.extend(U[:, int(k)] for k in keep)

    # Timing jitter commonly appears as temporal derivatives of the artifact.
    expanded: list[np.ndarray] = []
    for b in basis:
        expanded.append(b)
        if len(b) >= 3:
            d1 = np.gradient(b)
            expanded.append(d1)
    return expanded


def _artifact_similarity(vector: np.ndarray, basis: list[np.ndarray]) -> float:
    if not basis:
        return 0.0
    return max(abs(_safe_corr(vector, b)) for b in basis)


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
    *,
    n_sg: int | None = None,
    artifact_corr_threshold: float = 0.10,
    mean_corr_threshold: float = 0.10,
    mean_repeatability_threshold: float = 0.20,
    max_components: int = 6,
    groupwise: bool = False,
) -> np.ndarray:
    """Remove scanner-like high-frequency residual components section-wise.

    ``groupwise`` is accepted for API compatibility but deliberately ignored:
    this implementation always uses one PCA per temporal section.
    """
    del n_sg, groupwise

    vol_clean_signal = np.asarray(vol_clean_signal, dtype=np.float32)
    vol_noise_signal = np.asarray(vol_noise_signal, dtype=np.float32)
    onsets_up = np.asarray(onsets_up, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    subtracted = (vol_clean_signal - vol_noise_signal).astype(np.float32)
    sub_hpf70 = hpf_butter_1d(subtracted, srate_up, 70.0)
    noise_hpf70 = hpf_butter_1d(vol_noise_signal, srate_up, 70.0)

    sub_segs_hpf, valid_pca = extract_plain_segments(
        sub_hpf70, onsets_up, seg_len, indices=valid_idx
    )
    sub_segs_raw, valid_raw = extract_plain_segments(
        subtracted, onsets_up, seg_len, indices=valid_idx
    )
    noise_segs_hpf, valid_noise = extract_plain_segments(
        noise_hpf70, onsets_up, seg_len, indices=valid_idx
    )

    if not (
        np.array_equal(valid_pca, valid_raw)
        and np.array_equal(valid_pca, valid_noise)
    ):
        common = np.intersect1d(valid_pca, np.intersect1d(valid_raw, valid_noise))
        rp = {int(v): i for i, v in enumerate(valid_pca)}
        rr = {int(v): i for i, v in enumerate(valid_raw)}
        rn = {int(v): i for i, v in enumerate(valid_noise)}
        sub_segs_hpf = sub_segs_hpf[[rp[int(v)] for v in common]]
        sub_segs_raw = sub_segs_raw[[rr[int(v)] for v in common]]
        noise_segs_hpf = noise_segs_hpf[[rn[int(v)] for v in common]]
        valid_pca = common.astype(np.int64)

    clean_signal = vol_clean_signal.copy()
    clean_signal[scan_start_up:scan_stop_up] = subtracted[scan_start_up:scan_stop_up]
    if len(valid_pca) < 2:
        return clean_signal.astype(np.float32)

    scan_duration = (
        onsets_up[-1] + 2 * int(seg_len) + int(dtime_samp_up) - onsets_up[0]
    ) / float(srate_up)
    n_sections = max(1, int(round(scan_duration / float(time_section))))
    rows = np.arange(len(valid_pca), dtype=np.int64)
    rows_per_section = len(rows) / n_sections

    total_components = 0
    mean_removed_sections = 0

    for sec in range(n_sections):
        start_row = int(round(sec * rows_per_section))
        stop_row = int(round((sec + 1) * rows_per_section))
        sec_rows = rows[start_row:stop_row]
        if len(sec_rows) < 2:
            continue

        # samples x occurrences
        M = sub_segs_hpf[sec_rows].T.astype(np.float64)
        N = noise_segs_hpf[sec_rows].T.astype(np.float64)

        M0 = M - M.mean(axis=0, keepdims=True)
        mean_art = M0.mean(axis=1)
        residual = M0 - mean_art[:, None]
        basis = _artifact_basis(N)

        # Repeated section mean is removed only when the artifact templates
        # support it.  With no artifact reference, nothing is removed.
        within_rms = np.sqrt(np.mean(residual.T * residual.T, axis=1))
        denom = float(np.median(within_rms)) + 1e-30
        mean_repeatability = float(np.sqrt(np.mean(mean_art * mean_art)) / denom)
        mean_similarity = _artifact_similarity(mean_art, basis)
        remove_mean = (
            bool(basis)
            and mean_similarity >= float(mean_corr_threshold)
            and mean_repeatability >= float(mean_repeatability_threshold)
        )
        mean_to_remove = mean_art if remove_mean else np.zeros_like(mean_art)
        if remove_mean:
            mean_removed_sections += 1

        fitted = np.zeros_like(residual)
        selected: list[int] = []
        if basis and np.any(np.abs(residual) > 0):
            U, S, _ = np.linalg.svd(residual, full_matrices=False)
            eig = S * S
            var_pct = 100.0 * eig / (eig.sum() + 1e-30)
            eligible = np.flatnonzero(var_pct > float(var_threshold))
            if int(max_components) > 0:
                eligible = eligible[: int(max_components)]
            for k in eligible:
                if _artifact_similarity(U[:, int(k)], basis) >= float(artifact_corr_threshold):
                    selected.append(int(k))
            if selected:
                Uk = U[:, selected]
                fitted = Uk @ (Uk.T @ residual)

        total_components += len(selected)

        for j, row_idx in enumerate(sec_rows):
            slice_idx = int(valid_pca[row_idx])
            start = int(onsets_up[slice_idx])
            stop = start + int(seg_len)
            if 0 <= start and stop <= len(clean_signal):
                correction = fitted[:, j] + mean_to_remove
                clean_signal[start:stop] = (
                    sub_segs_raw[row_idx].astype(np.float64) - correction
                ).astype(np.float32)

        logger.debug(
            "Global PCA section=%d n=%d basis=%d mean_corr=%.3f "
            "mean_rep=%.3f mean_removed=%s components=%d",
            sec, len(sec_rows), len(basis), mean_similarity,
            mean_repeatability, remove_mean, len(selected),
        )

    logger.info(
        "Artifact-specific global PCA: %d sections, %d components removed, "
        "mean residual removed in %d sections (var>%.1f%%, corr>=%.2f)",
        n_sections, total_components, mean_removed_sections,
        var_threshold, artifact_corr_threshold,
    )
    return clean_signal.astype(np.float32)
