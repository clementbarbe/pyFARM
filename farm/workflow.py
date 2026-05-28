"""
farm pipeline orchestrator.

Calls each processing brick in the correct order.
Contains no complex scientific logic — only sequencing and data routing.
"""

import logging
import time
from pathlib import Path

import numpy as np

from farm.config import FARMConfig
from farm.utils.display import banner, ok, warn, info_table
from farm.utils.slices import build_slice_info

# ── I/O ──
from farm.io.loader import load_brainvision, select_channels
from farm.io.triggers import detect_volume_onsets

# ── Preprocessing ──
from farm.preprocessing.filters import hpf_fir, lpf_butter, apply_bandpass
from farm.preprocessing.resampling import upsample, downsample
from farm.preprocessing.trim import trim_to_scan

# ── Timing ──
from farm.timing.estimation import estimate_initial_timing
from farm.timing.optimization import optimize_global_timing, compute_slice_markers

# ── Correction bricks ──
from farm.alignment.phase_shift import (
    extract_aligned_segments, overwrite_segments,
)
from farm.templates.subtraction import build_artifact_templates
from farm.templates.zerofill import zero_fill_dtime
from farm.pca.cleanup import pca_cleanup

# ── Diagnostics & visualisation ──
from farm.diagnostics.segments import diagnose_segments
from farm.diagnostics.metrics import compute_rms_reduction

# ── Export ──
from farm.export.brainvision import export_brainvision
from farm.export.arrays import export_npz, export_mat
from farm.export.regressors import build_and_export_regressors

logger = logging.getLogger("farm.workflow")


# ═══════════════════════════════════════════════════════════════
# Private: per-channel FARM correction  (steps iii → vi)
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
) -> np.ndarray:
    """Run FARM steps (iii)–(vi) on a single upsampled channel.

    Returns the cleaned 1-D signal (float32).
    """
    signal = ch_signal.copy()

    # ── (iii) Phase alignment ────────────────────────────────
    aligned_segs, valid_idx = extract_aligned_segments(
        signal, onsets_up, seg_len, round_errors, padding=cfg.padding,
    )
    if len(valid_idx) == 0:
        logger.warning("Channel %s: no valid segments — skipping.", ch_name)
        return signal
    signal = overwrite_segments(signal, onsets_up, aligned_segs, valid_idx)

    if cfg.plot:
        diagnose_segments(signal, onsets_up, seg_len, ch_name,
                          "After (iii) phase-shift")

    # ── (v) Template subtraction ─────────────────────────────
    artifact_segs = build_artifact_templates(
        aligned_segs, valid_idx, slice_info,
        cfg.n_candidates, dtime_samp_up, seg_len,
    )
    artifact_signal = overwrite_segments(
        signal.copy(), onsets_up, artifact_segs, valid_idx,
    )

    if cfg.plot:
        sub_diag = signal.copy()
        sub_diag[scan_start_up:scan_stop_up] -= artifact_signal[scan_start_up:scan_stop_up]
        diagnose_segments(sub_diag, onsets_up, seg_len, ch_name,
                          "After (v) templates")
        del sub_diag

    # ── (iv) Zero-fill dtime ─────────────────────────────────
    signal_zf = zero_fill_dtime(
        signal, onsets_up, slice_info["last_slice_idx"],
        seg_len, dtime_samp_up,
    )
    artifact_zf = zero_fill_dtime(
        artifact_signal, onsets_up, slice_info["last_slice_idx"],
        seg_len, dtime_samp_up,
    )

    if cfg.plot:
        diagnose_segments(signal_zf, onsets_up, seg_len, ch_name,
                          "After (iv) zero-fill")

    # ── (vi) PCA cleanup ─────────────────────────────────────
    clean = pca_cleanup(
        signal_zf, artifact_zf,
        onsets_up, seg_len, valid_idx,
        srate_up, scan_start_up, scan_stop_up, dtime_samp_up,
        time_section=cfg.time_section,
        var_threshold=cfg.var_threshold,
    )

    if cfg.plot:
        diagnose_segments(clean, onsets_up, seg_len, ch_name,
                          "After (vi) PCA")

    return clean.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════

def run_pipeline(cfg: FARMConfig) -> dict:
    """Execute the full FARM denoising pipeline.

    Parameters
    ----------
    cfg : FARMConfig
        Fully populated pipeline configuration.

    Returns
    -------
    dict with keys:
        ``data_clean_full``, ``data_clean_crop``, ``data_hpf``,
        ``data_raw_trim``, ``data_full_raw``, ``ch_names``, ``ch_indices``,
        ``srate``, ``sdur``, ``dtime``, ``vol_onsets``, ``vol_onsets_abs``,
        ``s_trim``, ``e_trim``, ``n_vol``, ``raw_original``.
    """
    cfg.validate()
    t_total = time.time()
    basename = Path(cfg.vhdr_path).stem + "_FARM"

    if cfg.plot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.figsize": (14, 5), "figure.dpi": 100,
            "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
        })

    # ────────────────────────────────────────────────────────
    # 1. Load BrainVision data
    # ────────────────────────────────────────────────────────
    banner("LOADING DATA", "📂")
    raw = load_brainvision(cfg.vhdr_path)
    raw_original = raw.copy()
    srate = float(raw.info["sfreq"])

    # ────────────────────────────────────────────────────────
    # 2. Detect volume triggers
    # ────────────────────────────────────────────────────────
    banner("TRIGGER DETECTION", "🎯")
    vol_onsets, n_vol, ivi_stats = detect_volume_onsets(
        raw, cfg.trigger, cfg.tr, cfg.n_volumes,
    )
    if cfg.plot:
        from farm.visualization.timeseries import plot_ivi
        plot_ivi(ivi_stats, cfg.tr)

    # ────────────────────────────────────────────────────────
    # 3. Select EMG channels
    # ────────────────────────────────────────────────────────
    banner("CHANNEL SELECTION", "📡")
    data, ch_names, ch_indices = select_channels(raw, cfg.ch_regex)
    data_full_raw = data.copy()
    n_samples_full = data.shape[1]
    n_ch = data.shape[0]

    if cfg.plot:
        from farm.visualization.timeseries import plot_raw_channels
        plot_raw_channels(data, ch_names, srate, vol_onsets)

    # ────────────────────────────────────────────────────────
    # 4. Trim to scan boundaries
    # ────────────────────────────────────────────────────────
    banner("TRIM TO SCAN", "✂️")
    data, vol_onsets, vol_onsets_abs, s_trim, e_trim = trim_to_scan(
        data, vol_onsets, srate, cfg.tr,
    )
    data_raw_trim = data.copy()
    n_samples_orig = data.shape[1]

    # ────────────────────────────────────────────────────────
    # 5. High-pass filter 30 Hz
    # ────────────────────────────────────────────────────────
    banner("HPF 30 Hz", "🔊")
    t0 = time.time()
    data_before_hpf = data.copy()
    data = hpf_fir(data, srate, cfg.hpf_cutoff)
    data_hpf = data.copy()
    logger.info("HPF done in %.2f s", time.time() - t0)

    if cfg.plot:
        from farm.visualization.spectra import plot_psd_comparison
        plot_psd_comparison(
            data_before_hpf[0], data[0], srate, ch_names[0],
            "Before HPF", "After HPF 30 Hz",
        )
    del data_before_hpf

    # ────────────────────────────────────────────────────────
    # 6. Reference channel
    # ────────────────────────────────────────────────────────
    peak_abs = np.max(np.abs(data), axis=1)
    ref_ch = int(np.argmax(peak_abs))
    logger.info("Reference channel: %s (idx %d)", ch_names[ref_ch], ref_ch)

    # ────────────────────────────────────────────────────────
    # 7. Initial timing estimation
    # ────────────────────────────────────────────────────────
    banner("INITIAL TIMING", "⏱️")
    sdur_init, dtime_init, timing_diag = estimate_initial_timing(
        data[ref_ch], srate, vol_onsets, cfg.n_sg, n_vol,
    )
    info_table([
        ("sdur_init", f"{sdur_init * 1e3:.4f} ms"),
        ("dtime_init", f"{dtime_init * 1e3:.4f} ms"),
        ("TR reconstr.", f"{(cfg.n_sg * sdur_init + dtime_init) * 1e3:.4f} ms"),
    ])

    # ────────────────────────────────────────────────────────
    # 8. Upsample ×N
    # ────────────────────────────────────────────────────────
    banner(f"UPSAMPLE ×{cfg.interp_factor}", "📈")
    t0 = time.time()
    data_up, srate_up, vol_onsets_up = upsample(
        data, srate, vol_onsets, cfg.interp_factor,
    )
    logger.info("Upsample done in %.2f s — shape %s @ %.0f Hz",
                time.time() - t0, data_up.shape, srate_up)
    del data

    # ────────────────────────────────────────────────────────
    # 9. Global optimisation (Nelder-Mead)
    # ────────────────────────────────────────────────────────
    banner("GLOBAL TIMING OPTIMISATION", "🔬")
    sdur, dtime, opt_result = optimize_global_timing(
        data_up[ref_ch], srate_up, vol_onsets_up,
        sdur_init, dtime_init, cfg.n_sg, n_vol,
        padding=cfg.padding, window_size=cfg.window_size,
    )
    info_table([
        ("sdur", f"{sdur * 1e3:.6f} ms"),
        ("dtime", f"{dtime * 1e3:.6f} ms"),
        ("TR reconstr.", f"{(cfg.n_sg * sdur + dtime) * 1e3:.6f} ms"),
        ("Cost", f"{opt_result.fun:.6f}"),
        ("Converged", str(opt_result.success)),
    ])

    # ────────────────────────────────────────────────────────
    # 10. Slice markers
    # ────────────────────────────────────────────────────────
    banner("SLICE MARKERS", "📍")
    n_total = n_vol * cfg.n_sg
    onsets_up, round_errors, seg_len = compute_slice_markers(
        vol_onsets_up, sdur, dtime, srate_up, cfg.n_sg, n_vol,
    )
    slice_info = build_slice_info(n_total, cfg.n_sg, cfg.window_size)
    logger.info(
        "%d markers, seg_len=%d, good=%d",
        n_total, seg_len, len(slice_info["good_slice_idx"]),
    )

    if cfg.plot:
        from farm.visualization.timeseries import plot_slice_marker_diagnostics
        plot_slice_marker_diagnostics(onsets_up, round_errors, seg_len, cfg.n_sg)

    # ────────────────────────────────────────────────────────
    # 11. Per-channel FARM correction
    # ────────────────────────────────────────────────────────
    dtime_samp_up = int(round(dtime * srate_up))
    scan_start_up = int(onsets_up[0])
    scan_stop_up = min(data_up.shape[1],
                       int(onsets_up[-1] + seg_len + dtime_samp_up))

    for ch in range(n_ch):
        banner(f"CHANNEL {ch + 1}/{n_ch}: {ch_names[ch]}", "🔧")
        t0 = time.time()
        data_up[ch] = _correct_channel(
            data_up[ch], onsets_up, round_errors, seg_len,
            slice_info, scan_start_up, scan_stop_up,
            dtime_samp_up, cfg, srate_up, ch_names[ch],
        )
        logger.info("Channel %s done in %.2f s", ch_names[ch], time.time() - t0)

    # ────────────────────────────────────────────────────────
    # 12. Downsample + LPF
    # ────────────────────────────────────────────────────────
    banner("POST-PROCESSING", "📉")
    t0 = time.time()
    data_clean_crop = downsample(data_up, srate_up, cfg.interp_factor, n_samples_orig)
    del data_up
    data_clean_crop = lpf_butter(data_clean_crop, srate, cfg.lpf_cutoff)

    # Reconstruct full-length signal
    data_clean_full = data_full_raw.copy()
    data_clean_full[:, s_trim:e_trim] = data_clean_crop
    logger.info("Post-processing done in %.2f s", time.time() - t0)

    # ────────────────────────────────────────────────────────
    # 13. Final diagnostics
    # ────────────────────────────────────────────────────────
    banner("FINAL DIAGNOSTICS", "📊")
    rms_results = compute_rms_reduction(
        data_hpf, data_clean_crop, srate, ch_names, cfg.bandpass,
    )

    if cfg.plot:
        from farm.visualization.spectra import (
            plot_fft_power, plot_fft_before_after,
            plot_psd_welch_before_after, plot_spectrogram_comparison,
        )
        from farm.visualization.timeseries import (
            plot_session_comparison, plot_full_signal_overview,
        )
        from farm.visualization.carpet import plot_carpet_comparison

        for ch_i in range(n_ch):
            name = ch_names[ch_i]
            plot_fft_power(data_clean_crop[ch_i], srate, name)

            ts_b = apply_bandpass(data_hpf[ch_i], srate, cfg.bandpass)
            ts_a = apply_bandpass(data_clean_crop[ch_i], srate, cfg.bandpass)
            plot_fft_before_after(ts_b, ts_a, srate, name, sdur)
            plot_psd_welch_before_after(ts_b, ts_a, srate, name)
            plot_spectrogram_comparison(
                data_hpf[ch_i], data_clean_crop[ch_i], srate, name,
            )
            plot_session_comparison(
                data_hpf[ch_i], data_clean_crop[ch_i], srate, name, cfg.bandpass,
            )
            plot_carpet_comparison(
                data_hpf[ch_i], data_clean_crop[ch_i],
                srate, name, vol_onsets, sdur, cfg.n_sg, n_vol, cfg.bandpass,
            )
            plot_full_signal_overview(
                data_full_raw[ch_i], data_clean_full[ch_i],
                srate, name, s_trim, e_trim,
            )

    # ────────────────────────────────────────────────────────
    # 14. Export
    # ────────────────────────────────────────────────────────
    banner("EXPORT", "💾")

    export_brainvision(
        raw_original, data_clean_full, ch_indices,
        cfg.output_dir, basename,
    )
    export_npz(
        cfg.output_dir, basename,
        clean_full=data_clean_full,
        raw_trim=data_raw_trim,
        pca_clean=data_clean_crop,
        hpf_data=data_hpf,
        ch_names=np.array(ch_names),
        srate=srate,
        vol_onsets=vol_onsets,
        vol_onsets_abs=vol_onsets_abs,
        s_trim=s_trim, e_trim=e_trim,
        sdur=sdur, dtime=dtime, tr=cfg.tr, n_sg=cfg.n_sg,
    )
    export_mat(
        cfg.output_dir, basename,
        clean_full=data_clean_full,
        raw_trim=data_raw_trim,
        pca_clean=data_clean_crop,
        hpf_data=data_hpf,
        ch_names=np.array(ch_names, dtype=object),
        srate=srate,
        vol_onsets=vol_onsets,
        vol_onsets_abs=vol_onsets_abs,
        s_trim=s_trim, e_trim=e_trim,
        sdur=sdur, dtime=dtime, tr=cfg.tr, n_sg=cfg.n_sg,
        time_full=np.arange(n_samples_full) / srate,
        time_crop=np.arange(n_samples_orig) / srate,
    )

    # ────────────────────────────────────────────────────────
    # 15. EMG regressors
    # ────────────────────────────────────────────────────────
    banner("EMG REGRESSORS", "📉")
    all_regressors = build_and_export_regressors(
        data_clean_crop, srate, ch_names, n_vol, cfg.tr,
        cfg.bandpass, cfg.output_dir, basename,
    )

    # ────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "🏁")
    info_table([
        ("sdur", f"{sdur * 1e3:.4f} ms"),
        ("dtime", f"{dtime * 1e3:.4f} ms"),
        ("TR", f"{(cfg.n_sg * sdur + dtime) * 1e3:.4f} ms"),
        ("Total time", f"{time.time() - t_total:.1f} s"),
        ("Output", str(Path(cfg.output_dir).resolve())),
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
        "vol_onsets": vol_onsets,
        "vol_onsets_abs": vol_onsets_abs,
        "s_trim": s_trim,
        "e_trim": e_trim,
        "n_vol": n_vol,
        "raw_original": raw_original,
        "rms_results": rms_results,
        "regressors": all_regressors,
    }