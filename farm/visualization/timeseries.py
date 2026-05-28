"""Time-domain diagnostic plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_channels(
    data: np.ndarray,
    ch_names: list,
    srate: float,
    vol_onsets: np.ndarray,
    n_triggers_shown: int = 5,
) -> None:
    """Overview of raw channel signals with first trigger lines."""
    n_ch = data.shape[0]
    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    t_sec = np.arange(data.shape[1]) / srate
    for i, (ax, name) in enumerate(zip(axes, ch_names)):
        ax.plot(t_sec, data[i], linewidth=0.2, alpha=0.8)
        for vo in vol_onsets[:n_triggers_shown]:
            ax.axvline(vo / srate, color="r", alpha=0.3, lw=0.5)
        ax.set_ylabel(name)
        rms = float(np.sqrt(np.mean(data[i] ** 2)))
        ax.text(0.99, 0.95, f"RMS={rms:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="orange",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw data (first triggers in red)", fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_ivi(ivi_stats: dict, tr: float) -> None:
    """Inter-volume interval plot."""
    ivis = ivi_stats["ivis"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ivis * 1000, ".-", markersize=3)
    ax.axhline(tr * 1000, color="r", ls="--", alpha=0.5, label=f"TR={tr * 1000:.0f} ms")
    ax.set_xlabel("Volume #"); ax.set_ylabel("IVI (ms)")
    ax.set_title("Inter-volume interval"); ax.legend()
    fig.tight_layout()
    plt.show()


def plot_timing_diagnostics(diag: dict, srate: float, n_sg: int) -> None:
    """dtime scan curve and sdur/dtime distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # SV curve for first volume
    dtime_candidates = diag["dtime_candidates"]
    svs = diag.get("sv_first_vol")  # pre-computed or recalculated externally
    if svs is not None:
        axes[0].plot(dtime_candidates[: len(svs)], svs, linewidth=0.6)
        axes[0].set_xlabel("dtime (samples)")
        axes[0].set_ylabel("Sum of std")
        axes[0].set_title("SV curve — Volume 0")
    # Distributions
    axes[1].hist(diag["sdur_list"] * 1e3, bins=20, alpha=0.6, label="sdur (ms)")
    axes[1].hist(diag["dtime_list"] * 1e3, bins=20, alpha=0.6, label="dtime (ms)")
    axes[1].set_xlabel("ms"); axes[1].set_title("Distribution per volume")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def plot_slice_marker_diagnostics(
    onsets_up: np.ndarray,
    round_errors: np.ndarray,
    seg_len: int,
    n_sg: int,
) -> None:
    """Rounding-error histogram and inter-slice spacings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(round_errors, bins=50, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Δ(i) (upsampled samples)")
    axes[0].set_title("Rounding error distribution")
    axes[0].axvline(0, color="r", ls="--", alpha=0.5)
    spacings = np.diff(onsets_up[: n_sg * 3])
    axes[1].plot(spacings, ".-", markersize=2)
    axes[1].axhline(seg_len, color="r", ls="--", alpha=0.5,
                     label=f"seg_len={seg_len}")
    axes[1].set_xlabel("Segment #"); axes[1].set_ylabel("Spacing (samples)")
    axes[1].set_title("Inter-slice spacings (first 3 volumes)")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def plot_session_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_name: str,
    bandpass: tuple | None = None,
) -> None:
    """Full-session chronological comparison (filtered)."""
    from farm.preprocessing.filters import apply_bandpass

    N = len(data_before)
    stride = max(1, N // 20000)
    idx = np.arange(0, N, stride)
    t = idx / srate

    ts_b = apply_bandpass(data_before, srate, bandpass)[idx] if bandpass else data_before[idx]
    ts_a = apply_bandpass(data_after, srate, bandpass)[idx] if bandpass else data_after[idx]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(t, ts_b, alpha=0.45, lw=0.35, label="Before", color="#d62728")
    ax.plot(t, ts_a, alpha=0.95, lw=0.55, label="After FARM", color="#2ca02c")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title(f"Session comparison — {ch_name}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_full_signal_overview(
    data_raw: np.ndarray,
    data_clean: np.ndarray,
    srate: float,
    ch_name: str,
    s_trim: int,
    e_trim: int,
) -> None:
    """Full-length exported signal (raw vs. cleaned) with FARM region markers."""
    N = len(data_raw)
    stride = max(1, N // 15000)
    idx = np.arange(0, N, stride)
    t = idx / srate

    fig, axes = plt.subplots(2, 1, figsize=(16, 7))
    # Full view
    axes[0].plot(t, data_raw[idx], color="#d62728", lw=0.4, alpha=0.5, label="Raw")
    axes[0].plot(t, data_clean[idx], color="#2ca02c", lw=0.5, alpha=0.9, label="Cleaned")
    axes[0].axvline(s_trim / srate, color="blue", ls="--", alpha=0.5, lw=1, label="FARM region")
    axes[0].axvline(e_trim / srate, color="blue", ls="--", alpha=0.5, lw=1)
    axes[0].set_title(f"Full exported signal — {ch_name}")
    axes[0].set_xlabel("Time (s)"); axes[0].legend(fontsize=8)
    # Zoom
    zoom_n = min(int(20.0 * srate), N)
    t_z = np.arange(zoom_n) / srate
    axes[1].plot(t_z, data_raw[:zoom_n], color="#d62728", lw=0.6, alpha=0.7, label="Raw")
    axes[1].plot(t_z, data_clean[:zoom_n], color="#2ca02c", lw=0.8, alpha=0.95, label="Cleaned")
    axes[1].set_title(f"Zoom 20 s — {ch_name}")
    axes[1].set_xlabel("Time (s)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show()