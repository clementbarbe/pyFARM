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
from farm.preprocessing.filters import hpf_fir, lpf_butter, apply_bandpass
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
from farm.diagnostics.metrics import compute_rms_reduction

# ── Export ───────────────────────────────────────────────────
from farm.export.brainvision import export_brainvision
from farm.export.arrays import export_npz, export_mat
from farm.export.regressors import build_and_export_regressors


logger = logging.getLogger("farm.workflow")


# ═══════════════════════════════════════════════════════════════
# Private: per-channel FARM correction
# ═══════════════════════════════════════════════════════════════

def _correct_channel(
    ch_signal: np.ndarray,
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
    signal = np.asarray(ch_signal, dtype=np.float32).copy()

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

    # Replace valid slice segments by their phase-aligned equivalents.
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
        aligned_segments=aligned_segs,
        valid_idx=valid_idx,
        slice_info=slice_info,
        n_candidates=cfg.n_candidates,
        dtime_samp_up=dtime_samp_up,
        seg_len=seg_len,
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
            "After (iv) zero-fill",
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
        regressor information, QC metrics, and output metadata.
    """
    cfg.validate()

    t_total = time.time()

    run_name = Path(cfg.vhdr_path).stem + "_FARM"
    cfg.output_dir = str(Path(cfg.output_dir) / run_name)
    basename = run_name

    # ────────────────────────────────────────────────────────
    # Resolve figure output directory
    # ────────────────────────────────────────────────────────
    fig_dir: Path | None = None

    if cfg.figures_enabled:
        fig_dir = cfg.get_figures_dir()
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
    # 5. High-pass filter
    # ────────────────────────────────────────────────────────
    banner(f"HPF {cfg.hpf_cutoff:g} Hz", "🔊")

    t0 = time.time()

    data_before_hpf = data.copy()
    data = hpf_fir(data, srate, cfg.hpf_cutoff)
    data_hpf = data.copy()

    logger.info("HPF done in %.2f s", time.time() - t0)

    if fig_dir is not None:
        from farm.visualization.spectra import plot_psd_comparison

        plot_psd_comparison(
            data_before_hpf[0],
            data[0],
            srate,
            ch_names[0],
            fig_dir,
            "Before HPF",
            f"After HPF {cfg.hpf_cutoff:g} Hz",
        )

    del data_before_hpf

    # ────────────────────────────────────────────────────────
    # 6. Select reference channel
    # ────────────────────────────────────────────────────────
    banner("REFERENCE CHANNEL", "📡")

    peak_abs = np.max(np.abs(data), axis=1)
    ref_ch = int(np.argmax(peak_abs))

    logger.info(
        "Reference channel: %s (selected channel index %d; peak %.3e)",
        ch_names[ref_ch],
        ref_ch,
        peak_abs[ref_ch],
    )

    # ────────────────────────────────────────────────────────
    # 7. Coarse timing estimation
    # ────────────────────────────────────────────────────────
    banner("INITIAL TIMING", "⏱️")

    sdur_init, dtime_init, timing_diag = estimate_initial_timing(
        ref_signal=data[ref_ch],
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
    # 8. Upsample
    # ────────────────────────────────────────────────────────
    banner(f"UPSAMPLE ×{cfg.interp_factor}", "📈")

    t0 = time.time()

    data_up, srate_up, vol_onsets_up = upsample(
        data=data,
        srate=srate,
        vol_onsets=vol_onsets,
        factor=cfg.interp_factor,
    )

    logger.info(
        "Upsample done in %.2f s — shape %s @ %.3f Hz",
        time.time() - t0,
        data_up.shape,
        srate_up,
    )

    del data

    # ────────────────────────────────────────────────────────
    # 9. Global timing optimisation
    # ────────────────────────────────────────────────────────
    banner("GLOBAL TIMING OPTIMISATION", "🔬")

    sdur, dtime, opt_result = optimize_global_timing(
        ref_signal_up=data_up[ref_ch],
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
    scan_stop_up = data_up.shape[1]

    for ch in range(n_ch):
        banner(f"CHANNEL {ch + 1}/{n_ch}: {ch_names[ch]}", "🔧")

        t0 = time.time()

        data_up[ch] = _correct_channel(
            ch_signal=data_up[ch],
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
    # 12. Low-pass before downsampling
    # ────────────────────────────────────────────────────────
    banner("POST-PROCESSING", "📉")

    t0 = time.time()

    # Explicit low-pass filtering happens BEFORE downsampling.
    # This avoids residual high-frequency artifact being folded into lower
    # frequencies during the decimation step.
    data_up = lpf_butter(data_up, srate_up, cfg.lpf_cutoff)

    data_clean_crop = downsample(
        data_up=data_up,
        srate_up=srate_up,
        factor=cfg.interp_factor,
        target_length=n_samples_orig,
    )

    del data_up

    # Optional final LPF at original sampling frequency to remove tiny
    # resampling-edge residuals. It is normally harmless because data were
    # already low-passed at the upsampled rate.
    data_clean_crop = lpf_butter(
        data_clean_crop,
        srate,
        cfg.lpf_cutoff,
    )

    # Reconstruct the full-length EMG recording:
    # - outside the fMRI scan crop: untouched raw data;
    # - inside the crop: FARM-cleaned signal.
    data_clean_full = data_full_raw.copy()
    data_clean_full[:, s_trim:e_trim] = data_clean_crop

    logger.info("Post-processing done in %.2f s", time.time() - t0)

    # ────────────────────────────────────────────────────────
    # 13. Final diagnostics
    # ────────────────────────────────────────────────────────
    banner("FINAL DIAGNOSTICS", "📊")

    rms_results = compute_rms_reduction(
        data_before=data_hpf,
        data_after=data_clean_crop,
        srate=srate,
        ch_names=ch_names,
        bandpass=cfg.bandpass,
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

            plot_fft_power(
                data_clean_crop[ch_i],
                srate,
                name,
                fig_dir,
            )

            ts_before = apply_bandpass(
                data_hpf[ch_i],
                srate,
                cfg.bandpass,
            )

            ts_after = apply_bandpass(
                data_clean_crop[ch_i],
                srate,
                cfg.bandpass,
            )

            plot_fft_before_after(
                ts_before,
                ts_after,
                srate,
                name,
                fig_dir,
                sdur,
            )

            plot_psd_welch_before_after(
                ts_before,
                ts_after,
                srate,
                name,
                fig_dir,
            )

            plot_spectrogram_comparison(
                data_hpf[ch_i],
                data_clean_crop[ch_i],
                srate,
                name,
                fig_dir,
            )

            plot_session_comparison(
                data_hpf[ch_i],
                data_clean_crop[ch_i],
                srate,
                name,
                fig_dir,
                cfg.bandpass,
            )

            plot_carpet_comparison(
                data_before=data_hpf[ch_i],
                data_after=data_clean_crop[ch_i],
                srate=srate,
                ch_name=name,
                vol_onsets=vol_onsets,
                sdur=sdur,
                n_sg=cfg.n_sg,
                n_vol=n_vol,
                fig_dir=fig_dir,
                bandpass=cfg.bandpass,
            )

            plot_full_signal_overview(
                data_full_raw[ch_i],
                data_clean_full[ch_i],
                srate,
                name,
                s_trim,
                e_trim,
                fig_dir,
            )

    # ────────────────────────────────────────────────────────
    # 14. Export cleaned signal arrays
    # ────────────────────────────────────────────────────────
    banner("EXPORT", "💾")

    export_brainvision(
        raw_original=raw_original,
        data_clean_full=data_clean_full,
        ch_indices=ch_indices,
        output_dir=cfg.output_dir,
        basename=basename,
    )

    export_npz(
        cfg.output_dir,
        basename,
        clean_full=data_clean_full,
        raw_trim=data_raw_trim,
        pca_clean=data_clean_crop,
        hpf_data=data_hpf,
        ch_names=np.array(ch_names),
        srate=srate,
        vol_onsets=vol_onsets,
        vol_onsets_abs=vol_onsets_abs,
        volume_onsets_sec=volume_onsets_sec,
        s_trim=s_trim,
        e_trim=e_trim,
        sdur=sdur,
        dtime=dtime,
        tr=cfg.tr,
        observed_tr=observed_tr,
        n_sg=cfg.n_sg,
        n_vol=n_vol,
        zero_fill_gap_fraction=cfg.zero_fill_gap_fraction,
    )

    export_mat(
        cfg.output_dir,
        basename,
        clean_full=data_clean_full,
        raw_trim=data_raw_trim,
        pca_clean=data_clean_crop,
        hpf_data=data_hpf,
        ch_names=np.array(ch_names, dtype=object),
        srate=srate,
        vol_onsets=vol_onsets,
        vol_onsets_abs=vol_onsets_abs,
        volume_onsets_sec=volume_onsets_sec,
        s_trim=s_trim,
        e_trim=e_trim,
        sdur=sdur,
        dtime=dtime,
        tr=cfg.tr,
        observed_tr=observed_tr,
        n_sg=cfg.n_sg,
        n_vol=n_vol,
        zero_fill_gap_fraction=cfg.zero_fill_gap_fraction,
        time_full=np.arange(n_samples_full, dtype=np.float64) / srate,
        time_crop=np.arange(n_samples_orig, dtype=np.float64) / srate,
    )

    # ────────────────────────────────────────────────────────
    # 15. Build EMG regressors
    # ────────────────────────────────────────────────────────
    banner("EMG REGRESSORS", "📉")

    all_regressors = build_and_export_regressors(
        data_clean_crop=data_clean_crop,
        srate=srate,
        ch_names=ch_names,
        n_vol=n_vol,
        tr=cfg.tr,
        volume_onsets_sec=volume_onsets_sec,
        bandpass=cfg.bandpass,
        output_dir=cfg.output_dir,
        basename=basename,
        fig_dir=fig_dir,
        envelope_baseline=cfg.envelope_baseline,
        envelope_percentile=cfg.envelope_percentile,
        envelope_window_sec=cfg.envelope_window_sec,
        envelope_threshold_factor=cfg.envelope_threshold_factor,
        center_hrf=cfg.center_hrf,
        log_compress_gain=cfg.log_compress_gain,
    )

    # ────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "🏁")

    info_table([
        ("Complete volumes", f"{n_vol}"),
        ("Acquisition groups / volume", f"{cfg.n_sg}"),
        ("sdur", f"{sdur * 1e3:.4f} ms"),
        ("dtime", f"{dtime * 1e3:.4f} ms"),
        ("Observed TR", f"{observed_tr * 1e3:.4f} ms"),
        ("Reconstructed TR", f"{reconstructed_tr * 1e3:.4f} ms"),
        ("Zero-filled gap fraction", f"{cfg.zero_fill_gap_fraction:.2f}"),
        ("Total time", f"{time.time() - t_total:.1f} s"),
        ("Output", str(Path(cfg.output_dir).resolve())),
        ("Figures", str(fig_dir) if fig_dir else "disabled"),
    ])

    ok("Pipeline complete.")

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
        "regressors": all_regressors,
        "figures_dir": fig_dir,
        "output_dir": cfg.output_dir,
    }