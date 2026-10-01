"""
FARM pipeline orchestrator.

Calls each processing brick in the correct order.
Contains no complex scientific logic — only sequencing and data routing.

All diagnostic figures are saved as ``.png`` files to the configured
``figures_dir``. No interactive ``plt.show()`` is ever called.
"""

import logging
import time
from pathlib import Path

import numpy as np

from farm.config import FARMConfig
from farm.utils.display import banner, ok, info_table
from farm.utils.slices import build_slice_info

# ── I/O ──────────────────────────────────────────────────────
from farm.io.loader import load_brainvision, select_channels
from farm.io.triggers import detect_volume_onsets

# ── Preprocessing ────────────────────────────────────────────
from farm.preprocessing.filters import hpf_fir, apply_bandpass
from farm.preprocessing.resampling import upsample, downsample
from farm.preprocessing.trim import trim_to_scan

# ── Timing ───────────────────────────────────────────────────
from farm.timing.estimation import estimate_initial_timing
from farm.timing.optimization import (
    optimize_global_timing,
    compute_slice_markers,
)

# ── Correction bricks ────────────────────────────────────────
from farm.alignment.phase_shift import (
    extract_aligned_segments,
    overwrite_segments,
)
from farm.templates.subtraction import build_artifact_templates
from farm.templates.zerofill import zero_fill_dtime
from farm.pca.cleanup import pca_cleanup

# ── Diagnostics ──────────────────────────────────────────────
from farm.diagnostics.segments import diagnose_segments
from farm.diagnostics.metrics import (
    compute_rms_reduction,
    compute_scanner_locked_reduction,
)

# ── Export ───────────────────────────────────────────────────
from farm.export.brainvision import export_brainvision
from farm.export.metadata import export_metadata


logger = logging.getLogger("farm.workflow")


def _scanner_locked_reference_score(
    signal_1d: np.ndarray,
    vol_onsets: np.ndarray,
    tr_samples: int,
) -> float:
    """Ratio of trigger-locked mean energy to non-locked residual energy."""
    segments = []
    n = len(signal_1d)
    for onset in np.asarray(vol_onsets, dtype=np.int64):
        start = int(onset)
        stop = start + int(tr_samples)
        if 0 <= start and stop <= n:
            segments.append(signal_1d[start:stop])
    if len(segments) < 3:
        return 0.0
    X = np.asarray(segments, dtype=np.float64)
    template = np.median(X, axis=0)
    residual = X - template[None, :]
    locked_rms = float(np.sqrt(np.mean(template * template)))
    residual_rms = float(np.median(np.sqrt(np.mean(residual * residual, axis=1))))
    return locked_rms / (residual_rms + 1e-30)


def _select_reference_channel(
    timing_data: np.ndarray,
    vol_onsets: np.ndarray,
    srate: float,
    tr: float,
) -> tuple[int, np.ndarray]:
    """Select the channel with the most repeatable trigger-locked artifact."""
    tr_samples = max(8, int(round(float(tr) * float(srate))))
    scores = np.asarray([
        _scanner_locked_reference_score(ch, vol_onsets, tr_samples)
        for ch in np.asarray(timing_data)
    ], dtype=np.float64)
    return int(np.argmax(scores)), scores


# ═══════════════════════════════════════════════════════════════
# Private: per-channel FARM correction
# ═══════════════════════════════════════════════════════════════

def _correct_channel(
    ch_science_signal: np.ndarray,
    ch_artifact_signal: np.ndarray,
    onsets_up: np.ndarray,
    round_errors: np.ndarray,
    seg_len: int,
    slice_info: dict,
    scan_start_up: int,
    scan_stop_up: int,
    dtime_samp_up: int,
    cfg: FARMConfig,
    srate_up: float,
    ch_name: str = "",
    fig_dir: Path | None = None,
) -> np.ndarray:
    """Run FARM correction on a single upsampled channel.

    Processing:
        iii.  Fractional-sample phase alignment
        v.    Adaptive template construction/subtraction
        iv.   Inter-volume dead-time masking
        vi.   PCA residual cleanup

    Notes
    -----
    ``artifact_signal`` must start from zeros, not from a copy of ``signal``.
    Otherwise, in regions without a valid template, this would occur::

        subtracted = signal - artifact_signal = signal - signal = 0

    and valid EMG data could be removed during PCA cleanup.
    """
    signal = np.asarray(ch_science_signal, dtype=np.float32).copy()
    artifact_reference = np.asarray(ch_artifact_signal, dtype=np.float32).copy()

    # ── (iii) Fractional phase alignment ─────────────────────
    aligned_segs, valid_idx = extract_aligned_segments(
        signal_1d=signal,
        onsets=onsets_up,
        seg_len=seg_len,
        round_errors=round_errors,
        padding=cfg.padding,
    )

    if len(valid_idx) == 0:
        logger.warning(
            "Channel %s: no valid aligned segments; skipping correction.",
            ch_name,
        )
        return signal

    artifact_aligned, artifact_valid_idx = extract_aligned_segments(
        signal_1d=artifact_reference,
        onsets=onsets_up,
        seg_len=seg_len,
        round_errors=round_errors,
        padding=cfg.padding,
        indices=valid_idx,
    )

    if not np.array_equal(valid_idx, artifact_valid_idx):
        common = np.intersect1d(valid_idx, artifact_valid_idx)
        science_rows = {int(v): i for i, v in enumerate(valid_idx)}
        artifact_rows = {int(v): i for i, v in enumerate(artifact_valid_idx)}
        aligned_segs = aligned_segs[[science_rows[int(v)] for v in common]]
        artifact_aligned = artifact_aligned[[artifact_rows[int(v)] for v in common]]
        valid_idx = common.astype(np.int64)

    # Replace science slice segments by their phase-aligned equivalents.
    signal = overwrite_segments(
        signal_1d=signal,
        onsets=onsets_up,
        segments=aligned_segs,
        seg_indices=valid_idx,
    )

    if fig_dir is not None:
        diagnose_segments(
            signal,
            onsets_up,
            seg_len,
            ch_name,
            "After (iii) phase-shift",
            fig_dir,
        )

    # ── (v) Adaptive artifact-template construction ──────────
    artifact_segs = build_artifact_templates(
        aligned_segments=artifact_aligned,
        valid_idx=valid_idx,
        slice_info=slice_info,
        n_candidates=cfg.n_candidates,
        dtime_samp_up=dtime_samp_up,
        seg_len=seg_len,
        trim_fraction=cfg.template_trim_fraction,
        min_correlation=cfg.template_min_correlation,
        scale_bounds=cfg.template_scale_bounds,
    )

    # Important: artifact estimate is zero everywhere by default.
    # Only valid slice-segment regions receive a template estimate.
    artifact_signal = np.zeros_like(signal, dtype=np.float32)

    artifact_signal = overwrite_segments(
        signal_1d=artifact_signal,
        onsets=onsets_up,
        segments=artifact_segs,
        seg_indices=valid_idx,
    )

    if fig_dir is not None:
        sub_diag = signal.copy()
        sub_diag[scan_start_up:scan_stop_up] -= (
            artifact_signal[scan_start_up:scan_stop_up]
        )

        diagnose_segments(
            sub_diag,
            onsets_up,
            seg_len,
            ch_name,
            "After (v) templates",
            fig_dir,
        )
        del sub_diag

    # ── (iv) Zero-fill only true inter-volume gaps ───────────
    signal_zf = zero_fill_dtime(
        signal=signal,
        onsets_up=onsets_up,
        last_slice_idx=slice_info["last_slice_idx"],
        seg_len=seg_len,
        dtime_samp_up=dtime_samp_up,
        gap_fraction=cfg.zero_fill_gap_fraction,
    )

    artifact_zf = zero_fill_dtime(
        signal=artifact_signal,
        onsets_up=onsets_up,
        last_slice_idx=slice_info["last_slice_idx"],
        seg_len=seg_len,
        dtime_samp_up=dtime_samp_up,
        gap_fraction=cfg.zero_fill_gap_fraction,
    )

    if fig_dir is not None:
        diagnose_segments(
            signal_zf,
            onsets_up,
            seg_len,
            ch_name,
            f"After (iv) gap mask ({cfg.zero_fill_gap_fraction:.2f})",
            fig_dir,
        )

    # ── (vi) PCA residual cleanup ────────────────────────────
    clean = pca_cleanup(
        vol_clean_signal=signal_zf,
        vol_noise_signal=artifact_zf,
        onsets_up=onsets_up,
        seg_len=seg_len,
        valid_idx=valid_idx,
        srate_up=srate_up,
        scan_start_up=scan_start_up,
        scan_stop_up=scan_stop_up,
        dtime_samp_up=dtime_samp_up,
        time_section=cfg.time_section,
        var_threshold=cfg.var_threshold,
        n_sg=cfg.n_sg,
        artifact_corr_threshold=cfg.pca_artifact_corr_threshold,
        mean_corr_threshold=cfg.pca_mean_corr_threshold,
        mean_repeatability_threshold=cfg.pca_mean_repeatability_threshold,
        max_components=cfg.pca_max_components,
        groupwise=cfg.pca_groupwise,
    )

    if fig_dir is not None:
        diagnose_segments(
            clean,
            onsets_up,
            seg_len,
            ch_name,
            "After (vi) PCA",
            fig_dir,
        )

    return clean.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════

def run_pipeline(cfg: FARMConfig) -> dict:
    """Execute the full FARM EMG-fMRI denoising pipeline.

    Parameters
    ----------
    cfg
        Fully populated FARM configuration.

    Returns
    -------
    dict
        Contains cleaned data, timing parameters, trigger positions,
        QC metrics, and output metadata.
    """
    cfg.validate()

    t_total = time.time()

    basename = Path(cfg.vhdr_path).stem + cfg.output_suffix
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────
    # Resolve figure output directory
    # ────────────────────────────────────────────────────────
    fig_dir: Path | None = None

    if cfg.figures_enabled:
        fig_dir = cfg.get_figures_dir(basename)
        logger.info("Figures will be saved to: %s", fig_dir)

    # ────────────────────────────────────────────────────────
    # 1. Load BrainVision data
    # ────────────────────────────────────────────────────────
    banner("LOADING DATA", "📂")

    raw = load_brainvision(cfg.vhdr_path)
    raw_original = raw.copy()

    srate = float(raw.info["sfreq"])
    cfg.validate_sampling_rate(srate)

    # ────────────────────────────────────────────────────────
    # 2. Detect volume triggers
    # ────────────────────────────────────────────────────────
    banner("TRIGGER DETECTION", "🎯")

    vol_onsets, n_vol, ivi_stats = detect_volume_onsets(
        raw=raw,
        trigger=cfg.trigger,
        tr=cfg.tr,
        n_volumes=cfg.n_volumes,
        drop_last=cfg.drop_last_volume,
    )

    if cfg.drop_last_volume:
        info_table([
            (
                "⚠️ drop_last_volume",
                "ON — last trigger removed (manual stop assumed)",
            ),
            ("Volumes retained", f"{n_vol}"),
        ])

    info_table([
        ("Volumes", f"{n_vol}"),
        ("IVI mean", f"{ivi_stats['mean'] * 1e3:.3f} ms"),
        ("IVI std", f"{ivi_stats['std'] * 1e3:.3f} ms"),
        ("IVI jitter max", f"{ivi_stats['jitter_max'] * 1e6:.1f} µs"),
        ("Configured TR", f"{cfg.tr * 1e3:.3f} ms"),
    ])

    if fig_dir is not None:
        from farm.visualization.timeseries import plot_ivi
        plot_ivi(ivi_stats, cfg.tr, fig_dir)

    # ────────────────────────────────────────────────────────
    # 3. Select EMG channels
    # ────────────────────────────────────────────────────────
    banner("CHANNEL SELECTION", "📡")

    data, ch_names, ch_indices = select_channels(raw, cfg.ch_regex)

    data_full_raw = data.copy()
    n_samples_full = data.shape[1]
    n_ch = data.shape[0]

    if fig_dir is not None:
        from farm.visualization.timeseries import plot_raw_channels
        plot_raw_channels(data, ch_names, srate, vol_onsets, fig_dir)

    # ────────────────────────────────────────────────────────
    # 4. Trim to complete scan volumes
    # ────────────────────────────────────────────────────────
    banner("TRIM TO SCAN", "✂️")

    (
        data,
        vol_onsets,
        vol_onsets_abs,
        s_trim,
        e_trim,
        n_vol,
    ) = trim_to_scan(
        data=data,
        vol_onsets=vol_onsets,
        srate=srate,
        tr=cfg.tr,
    )

    data_raw_trim = data.copy()
    n_samples_orig = data.shape[1]

    # Onsets are now relative to the cropped data array.
    # These are the correct real times to sample the fMRI regressors.
    volume_onsets_sec = vol_onsets.astype(np.float64) / srate

    info_table([
        ("Complete volumes", f"{n_vol}"),
        (
            "Working region",
            f"samples {s_trim}–{e_trim} ({n_samples_orig / srate:.1f} s)",
        ),
        (
            "Full signal",
            f"{n_samples_full:,} samples ({n_samples_full / srate:.1f} s)",
        ),
        ("First relative onset", f"{volume_onsets_sec[0]:.6f} s"),
        ("Last relative onset", f"{volume_onsets_sec[-1]:.6f} s"),
        ("Guarantee", "Every retained volume has a full TR of data"),
    ])

    # ────────────────────────────────────────────────────────
    # 5. Separate science and artifact-estimation branches
    # ────────────────────────────────────────────────────────
    banner("SCIENCE / ARTIFACT BRANCHES", "🔊")

    t0 = time.time()

    # Keep an untouched science branch, but derive scanner timing/templates
    # from the same 30-Hz HPF branch used by the proven legacy implementation.
    # The final EMG band is imposed only once, after FARM correction.
    data_science = data.copy()
    data_artifact = hpf_fir(data_science, srate, cfg.artifact_hpf_cutoff)
    if np.isclose(cfg.timing_hpf_cutoff, cfg.artifact_hpf_cutoff):
        data_timing = data_artifact
    else:
        data_timing = hpf_fir(data_science, srate, cfg.timing_hpf_cutoff)
    data_hpf = data_artifact.copy()

    logger.info(
        "Branches prepared in %.2f s (artifact HPF %.1f Hz, timing HPF %.1f Hz)",
        time.time() - t0, cfg.artifact_hpf_cutoff, cfg.timing_hpf_cutoff,
    )

    if fig_dir is not None:
        from farm.visualization.spectra import plot_psd_comparison

        plot_psd_comparison(
            data_science[0],
            data_artifact[0],
            srate,
            ch_names[0],
            fig_dir,
            "Science branch (unfiltered)",
            f"Artifact branch HPF {cfg.artifact_hpf_cutoff:g} Hz",
        )

    # ────────────────────────────────────────────────────────
    # 6. Select reference channel by scanner-locked repeatability
    # ────────────────────────────────────────────────────────
    banner("REFERENCE CHANNEL", "📡")

    # Restore the reference criterion that produced the stable timing on the
    # validation run: largest artifact excursion on the HPF branch.  The
    # repeatability score is still logged as a diagnostic, but it no longer
    # chooses a different timing minimum by itself.
    peak_abs = np.max(np.abs(data_timing), axis=1)
    ref_ch = int(np.argmax(peak_abs))
    _, ref_scores = _select_reference_channel(
        timing_data=data_timing,
        vol_onsets=vol_onsets,
        srate=srate,
        tr=cfg.tr,
    )
    logger.info(
        "Reference channel: %s (index %d; peak %.3e; locked scores=%s)",
        ch_names[ref_ch], ref_ch, peak_abs[ref_ch],
        np.array2string(ref_scores, precision=3),
    )

    # ────────────────────────────────────────────────────────
    # 7. Coarse timing estimation
    # ────────────────────────────────────────────────────────
    banner("INITIAL TIMING", "⏱️")

    sdur_init, dtime_init, timing_diag = estimate_initial_timing(
        ref_signal=data_timing[ref_ch],
        srate=srate,
        vol_onsets=vol_onsets,
        n_sg=cfg.n_sg,
        n_vol=n_vol,
    )

    del timing_diag

    info_table([
        ("Acquisition groups / volume", f"{cfg.n_sg}"),
        ("sdur_init", f"{sdur_init * 1e3:.4f} ms"),
        ("dtime_init", f"{dtime_init * 1e3:.4f} ms"),
        (
            "TR reconstr.",
            f"{(cfg.n_sg * sdur_init + dtime_init) * 1e3:.4f} ms",
        ),
    ])

    # ────────────────────────────────────────────────────────
    # 8. Upsample science + artifact branches
    # ────────────────────────────────────────────────────────
    banner(f"UPSAMPLE ×{cfg.interp_factor}", "📈")

    t0 = time.time()

    data_science_up, srate_up, vol_onsets_up = upsample(
        data=data_science,
        srate=srate,
        vol_onsets=vol_onsets,
        factor=cfg.interp_factor,
    )
    data_artifact_up, srate_artifact_up, vol_onsets_artifact_up = upsample(
        data=data_artifact,
        srate=srate,
        vol_onsets=vol_onsets,
        factor=cfg.interp_factor,
    )
    if not (
        np.isclose(srate_up, srate_artifact_up)
        and np.array_equal(vol_onsets_up, vol_onsets_artifact_up)
    ):
        raise RuntimeError("Upsampled branches are not temporally aligned")

    timing_ref_up = data_artifact_up[ref_ch]

    logger.info(
        "Upsample done in %.2f s — science %s, artifact %s @ %.3f Hz",
        time.time() - t0,
        data_science_up.shape,
        data_artifact_up.shape,
        srate_up,
    )

    del data, data_science, data_artifact, data_timing

    # ────────────────────────────────────────────────────────
    # 9. Global timing optimisation
    # ────────────────────────────────────────────────────────
    banner("GLOBAL TIMING OPTIMISATION", "🔬")

    sdur, dtime, opt_result = optimize_global_timing(
        ref_signal_up=timing_ref_up,
        srate_up=srate_up,
        vol_onsets_up=vol_onsets_up,
        sdur_init=sdur_init,
        dtime_init=dtime_init,
        n_sg=cfg.n_sg,
        n_vol=n_vol,
        padding=cfg.padding,
        window_size=cfg.window_size,
    )

    observed_tr = float(np.median(np.diff(vol_onsets))) / srate
    reconstructed_tr = cfg.n_sg * sdur + dtime

    info_table([
        ("sdur", f"{sdur * 1e3:.6f} ms"),
        ("dtime", f"{dtime * 1e3:.6f} ms"),
        ("Observed TR", f"{observed_tr * 1e3:.6f} ms"),
        ("TR reconstruction", f"{reconstructed_tr * 1e3:.6f} ms"),
        ("Timing cost", f"{opt_result.fun:.6f}"),
        ("Converged", str(opt_result.success)),
    ])

    # ────────────────────────────────────────────────────────
    # 10. Trigger-anchored slice markers
    # ────────────────────────────────────────────────────────
    banner("SLICE MARKERS", "📍")

    n_total = n_vol * cfg.n_sg

    onsets_up, round_errors, seg_len = compute_slice_markers(
        vol_onsets_up=vol_onsets_up,
        sdur=sdur,
        dtime=dtime,
        srate_up=srate_up,
        n_sg=cfg.n_sg,
        n_vol=n_vol,
    )

    slice_info = build_slice_info(
        n_total=n_total,
        n_sg=cfg.n_sg,
        window_size=cfg.window_size,
    )

    logger.info(
        "Slice markers: %d total, seg_len=%d samples, "
        "%d timing-good slices, %d last-group slices",
        n_total,
        seg_len,
        len(slice_info["good_slice_idx"]),
        len(slice_info["last_slice_idx"]),
    )

    if fig_dir is not None:
        from farm.visualization.timeseries import plot_slice_marker_diagnostics

        plot_slice_marker_diagnostics(
            onsets_up=onsets_up,
            round_errors=round_errors,
            seg_len=seg_len,
            n_sg=cfg.n_sg,
            fig_dir=fig_dir,
        )

    # ────────────────────────────────────────────────────────
    # 11. Per-channel FARM correction
    # ────────────────────────────────────────────────────────
    banner("FARM CORRECTION", "🔧")

    dtime_samp_up = int(round(dtime * srate_up))

    # Crop starts at the first retained volume trigger, so scan start is
    # generally zero. Using whole crop boundaries prevents final volume
    # samples from being silently excluded from PCA residual subtraction.
    scan_start_up = 0
    scan_stop_up = data_science_up.shape[1]

    for ch in range(n_ch):
        banner(f"CHANNEL {ch + 1}/{n_ch}: {ch_names[ch]}", "🔧")

        t0 = time.time()

        data_science_up[ch] = _correct_channel(
            ch_science_signal=data_science_up[ch],
            ch_artifact_signal=data_artifact_up[ch],
            onsets_up=onsets_up,
            round_errors=round_errors,
            seg_len=seg_len,
            slice_info=slice_info,
            scan_start_up=scan_start_up,
            scan_stop_up=scan_stop_up,
            dtime_samp_up=dtime_samp_up,
            cfg=cfg,
            srate_up=srate_up,
            ch_name=ch_names[ch],
            fig_dir=fig_dir,
        )

        logger.info(
            "Channel %s done in %.2f s",
            ch_names[ch],
            time.time() - t0,
        )

    # ────────────────────────────────────────────────────────
    # 12. Downsample pure FARM output
    # ────────────────────────────────────────────────────────
    banner("POST-PROCESSING", "📉")

    t0 = time.time()

    # resample_poly supplies the anti-alias filtering needed for decimation.
    # No analysis-band HPF/LPF is imposed here: the exported waveform remains
    # the broadband FARM result.  This cleanly separates denoising from later
    # EMG feature/regressor construction.
    data_clean_crop = downsample(
        data_up=data_science_up,
        srate_up=srate_up,
        factor=cfg.interp_factor,
        target_length=n_samples_orig,
    )
    del data_science_up, data_artifact_up

    # Reconstruct the full recording.  Outside the fMRI scan crop, data are
    # untouched; inside it, only the selected EMG channels are FARM-cleaned.
    data_clean_full = data_full_raw.copy()
    data_clean_full[:, s_trim:e_trim] = data_clean_crop

    # QC is intentionally performed in a configured EMG band, but this branch
    # is diagnostic only and never replaces the exported broadband data.
    data_before_qc = apply_bandpass(data_raw_trim, srate, cfg.qc_bandpass)
    data_after_qc = apply_bandpass(data_clean_crop, srate, cfg.qc_bandpass)

    logger.info(
        "Post-processing done in %.2f s — broadband FARM output preserved; "
        "QC evaluated at %.1f–%.1f Hz",
        time.time() - t0, cfg.qc_bandpass[0], cfg.qc_bandpass[1],
    )

    # ────────────────────────────────────────────────────────
    # 13. Final diagnostics
    # ────────────────────────────────────────────────────────
    banner("FINAL DIAGNOSTICS", "📊")

    rms_results = compute_rms_reduction(
        data_before=data_before_qc,
        data_after=data_after_qc,
        srate=srate,
        ch_names=ch_names,
        bandpass=None,
    )
    scanner_locked_results = compute_scanner_locked_reduction(
        data_before=data_before_qc,
        data_after=data_after_qc,
        srate=srate,
        ch_names=ch_names,
        vol_onsets=vol_onsets,
        tr=cfg.tr,
        bandpass=None,
    )

    if fig_dir is not None:
        from farm.visualization.spectra import (
            plot_fft_power,
            plot_fft_before_after,
            plot_psd_welch_before_after,
            plot_spectrogram_comparison,
        )
        from farm.visualization.timeseries import (
            plot_session_comparison,
            plot_full_signal_overview,
        )
        from farm.visualization.carpet import plot_carpet_comparison

        for ch_i in range(n_ch):
            name = ch_names[ch_i]

            # Main artifact-removal diagnostics use identical QC preprocessing
            # before/after.  This prevents the plot itself from introducing an
            # apparent spectral difference.
            plot_fft_power(data_after_qc[ch_i], srate, name, fig_dir)
            plot_fft_before_after(
                data_before_qc[ch_i], data_after_qc[ch_i],
                srate, name, fig_dir, sdur,
            )
            plot_psd_welch_before_after(
                data_before_qc[ch_i], data_after_qc[ch_i],
                srate, name, fig_dir,
            )
            plot_spectrogram_comparison(
                data_before_qc[ch_i], data_after_qc[ch_i],
                srate, name, fig_dir,
            )
            plot_session_comparison(
                data_before_qc[ch_i], data_after_qc[ch_i],
                srate, name, fig_dir, bandpass=None,
            )
            plot_carpet_comparison(
                data_before=data_before_qc[ch_i],
                data_after=data_after_qc[ch_i],
                srate=srate, ch_name=name, vol_onsets=vol_onsets,
                sdur=sdur, n_sg=cfg.n_sg, n_vol=n_vol,
                fig_dir=fig_dir, bandpass=None,
            )
            plot_full_signal_overview(
                data_full_raw[ch_i], data_clean_full[ch_i],
                srate, name, s_trim, e_trim, fig_dir,
            )

    # ────────────────────────────────────────────────────────
    # 14. Export pure denoised BrainVision + compact sidecar
    # ────────────────────────────────────────────────────────
    banner("EXPORT", "💾")

    brainvision_path = export_brainvision(
        raw_original=raw_original,
        data_clean_full=data_clean_full,
        ch_indices=ch_indices,
        output_dir=cfg.output_dir,
        basename=basename,
    )

    metadata = {
        "format_version": 1,
        "kind": "pyFARM-denoise",
        "config": dict(cfg.__dict__),
        "input_vhdr": str(Path(cfg.vhdr_path).resolve()),
        "output_vhdr": brainvision_path,
        "output_is_broadband_farm": True,
        "selected_channels": ch_names,
        "sampling_rate_hz": srate,
        "tr_seconds": cfg.tr,
        "n_slices": cfg.n_slices,
        "mb_factor": cfg.mb_factor,
        "n_slice_groups": cfg.n_sg,
        "trigger": cfg.trigger,
        "n_volumes": n_vol,
        "drop_last_volume": cfg.drop_last_volume,
        "scan_start_sample": s_trim,
        "scan_stop_sample_exclusive": e_trim,
        "volume_onsets_samples_full": vol_onsets_abs,
        "volume_onsets_samples_crop": vol_onsets,
        "sdur_seconds": sdur,
        "dtime_seconds": dtime,
        "observed_tr_seconds": observed_tr,
        "reconstructed_tr_seconds": reconstructed_tr,
        "zero_fill_gap_fraction": cfg.zero_fill_gap_fraction,
        "qc_bandpass_hz": cfg.qc_bandpass,
        "rms_reduction": rms_results,
        "scanner_locked_reduction": scanner_locked_results,
    }
    metadata_path = export_metadata(cfg.output_dir, basename, metadata)
    logger.info("Denoising sidecar: %s", metadata_path)

    # ────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────
    banner("DENOISING COMPLETE", "🏁")

    info_table([
        ("Complete volumes", f"{n_vol}"),
        ("Acquisition groups / volume", f"{cfg.n_sg}"),
        ("sdur", f"{sdur * 1e3:.4f} ms"),
        ("dtime", f"{dtime * 1e3:.4f} ms"),
        ("Observed TR", f"{observed_tr * 1e3:.4f} ms"),
        ("Reconstructed TR", f"{reconstructed_tr * 1e3:.4f} ms"),
        ("Zero-filled gap fraction", f"{cfg.zero_fill_gap_fraction:.2f}"),
        ("Exported signal", "broadband FARM (no regressor preprocessing)"),
        ("Total time", f"{time.time() - t_total:.1f} s"),
        ("Output", str(Path(cfg.output_dir).resolve())),
        ("Figures", str(fig_dir) if fig_dir else "disabled"),
    ])

    ok("Denoising complete.")

    return {
        "data_clean_full": data_clean_full,
        "data_clean_crop": data_clean_crop,
        "data_hpf": data_hpf,
        "data_raw_trim": data_raw_trim,
        "data_full_raw": data_full_raw,
        "ch_names": ch_names,
        "ch_indices": ch_indices,
        "srate": srate,
        "sdur": sdur,
        "dtime": dtime,
        "observed_tr": observed_tr,
        "vol_onsets": vol_onsets,
        "vol_onsets_abs": vol_onsets_abs,
        "volume_onsets_sec": volume_onsets_sec,
        "s_trim": s_trim,
        "e_trim": e_trim,
        "n_vol": n_vol,
        "n_sg": cfg.n_sg,
        "raw_original": raw_original,
        "rms_results": rms_results,
        "scanner_locked_results": scanner_locked_results,
        "figures_dir": fig_dir,
        "output_dir": cfg.output_dir,
        "brainvision_path": brainvision_path,
        "metadata_path": metadata_path,
    }
