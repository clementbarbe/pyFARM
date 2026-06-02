"""Regressor diagnostic plots.

- ``plot_envelope_correction`` — baseline correction diagnostic.
- ``plot_regressor`` — single-axis overview (MATLAB farm_plot_regressor style).
- ``plot_regressor_panels`` — multi-panel detailed view.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plotting import savefig


_COLORS = [
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
]


# ═══════════════════════════════════════════════════════════════
#  Envelope baseline correction diagnostic
# ═══════════════════════════════════════════════════════════════

def plot_envelope_correction(
    envelope_raw: np.ndarray,
    envelope_corrected: np.ndarray,
    correction_info: dict,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
) -> None:
    """Three-panel diagnostic showing the baseline correction effect."""
    method = correction_info.get("method", "none")
    if method == "none":
        return

    N = len(envelope_raw)
    t = np.arange(N) / srate
    stride = max(1, N // 25000)
    sl = slice(None, None, stride)

    baseline = correction_info.get("baseline")
    threshold_value = correction_info.get("threshold_value", 0.0)
    noise_std = correction_info.get("noise_std", 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)

    # Panel 1: original + baseline + threshold
    ax = axes[0]
    ax.plot(t[sl], envelope_raw[sl],
            color="black", lw=0.4, alpha=0.7, label="Original envelope")
    if baseline is not None:
        ax.plot(t[sl], baseline[sl],
                color="#d62728", lw=1.5, alpha=0.9, label="Estimated baseline")
        ax.fill_between(t[sl], 0, baseline[sl], color="#d62728", alpha=0.08)
    if threshold_value > 0 and baseline is not None:
        thresh_line = np.clip(baseline + threshold_value, 0, 1)
        ax.plot(t[sl], thresh_line[sl],
                color="#FF9800", lw=1.0, ls="--", alpha=0.8,
                label=f"Threshold (baseline + {threshold_value:.4f})")
    ax.set_ylabel("Amplitude [0, 1]")
    ax.set_title(f"Envelope baseline correction — {ch_name} — method: {method}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.2)

    # Panel 2: corrected
    ax = axes[1]
    ax.plot(t[sl], envelope_raw[sl],
            color="#d62728", lw=0.3, alpha=0.3, label="Before correction")
    ax.plot(t[sl], envelope_corrected[sl],
            color="#2ca02c", lw=0.5, alpha=0.9, label="After correction")
    ax.set_ylabel("Amplitude [0, 1]")
    ax.set_title(f"Corrected envelope — {ch_name}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.2)
    pct_zero = 100.0 * np.mean(envelope_corrected == 0)
    ax.text(
        0.01, 0.95,
        f"Samples at zero: {pct_zero:.1f}%  |  "
        f"σ_noise: {noise_std:.4f}  |  threshold: {threshold_value:.4f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Panel 3: zoom on transition
    ax = axes[2]
    zoom_half = int(min(10.0 * srate, N // 4))
    zoom_center = _find_transition(envelope_corrected, srate)
    z_start = max(0, zoom_center - zoom_half)
    z_stop = min(N, zoom_center + zoom_half)
    z_sl = slice(z_start, z_stop)
    t_z = t[z_sl]
    ax.plot(t_z, envelope_raw[z_sl],
            color="#d62728", lw=0.6, alpha=0.5, label="Before")
    ax.plot(t_z, envelope_corrected[z_sl],
            color="#2ca02c", lw=1.0, alpha=0.9, label="After")
    if baseline is not None:
        ax.plot(t_z, baseline[z_sl],
                color="#d62728", lw=1.2, ls="--", alpha=0.6, label="Baseline")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude [0, 1]")
    ax.set_title(f"Zoom on quiet↔active transition — {ch_name}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_envelope_correction")


def _find_transition(
    envelope: np.ndarray, srate: float, min_jump: float = 0.2,
) -> int:
    """Find a sample index near a quiet→active transition."""
    from scipy.ndimage import uniform_filter1d
    block_len = max(3, int(srate * 0.5))
    smoothed = uniform_filter1d(envelope.astype(np.float64), block_len)
    diff = np.diff(smoothed)
    if len(diff) == 0:
        return len(envelope) // 2
    candidates = np.where(diff > min_jump * diff.max())[0]
    if len(candidates) == 0:
        return int(np.argmax(diff))
    margin = int(10 * srate)
    for c in candidates:
        if margin < c < len(envelope) - margin:
            return int(c)
    return int(candidates[len(candidates) // 2])


# ═══════════════════════════════════════════════════════════════
#  Single-axis regressor overview
# ═══════════════════════════════════════════════════════════════

def plot_regressor(
    reginfo: dict,
    ch_name: str,
    fig_dir: Path | None,
    envelope: np.ndarray | None = None,
    srate_envelope: float | None = None,
    centered: bool = True,
) -> None:
    """Single-axis overview of all regressor traces for one channel.

    Reproduces the MATLAB ``farm_plot_regressor`` layout.

    Parameters
    ----------
    centered : bool — if *True*, adds a zero line and adapts the
        y-axis label to reflect that baseline = 0.
    """
    t_conv = reginfo["time_conv"]
    t_reg = reginfo["time_reg"]
    stride = max(1, len(t_conv) // 30000)
    sc = slice(None, None, stride)

    fig, ax = plt.subplots(figsize=(18, 6))

    # Envelope input
    if envelope is not None and srate_envelope is not None:
        t_env = np.arange(len(envelope)) / srate_envelope
        stride_e = max(1, len(t_env) // 30000)
        ax.plot(t_env[::stride_e], envelope[::stride_e],
                ls="-", color="black", lw=0.6, alpha=0.5, label="envelope (in)")

    # conv / reg
    ax.plot(t_conv[sc], reginfo["conv"][sc],
            ls="-", color=_COLORS[1], lw=0.7, label="conv")
    ax.plot(t_reg, reginfo["reg"],
            ls="-.", color=_COLORS[1], lw=1.2, marker=".", ms=3, label="reg")

    # dconv / dreg
    ax.plot(t_conv[sc], reginfo["dconv"][sc],
            ls="-", color=_COLORS[2], lw=0.7, label="dconv")
    ax.plot(t_reg, reginfo["dreg"],
            ls="-.", color=_COLORS[2], lw=1.2, marker=".", ms=3, label="dreg")

    # log_conv / log_reg
    ax.plot(t_conv[sc], reginfo["log_conv"][sc],
            ls="-", color=_COLORS[3], lw=0.7, label="log_conv")
    ax.plot(t_reg, reginfo["log_reg"],
            ls="-.", color=_COLORS[3], lw=1.2, marker=".", ms=3, label="log_reg")

    # dlog_conv / dlog_reg
    ax.plot(t_conv[sc], reginfo["dlog_conv"][sc],
            ls="-", color=_COLORS[4], lw=0.7, label="dlog_conv")
    ax.plot(t_reg, reginfo["dlog_reg"],
            ls="-.", color=_COLORS[4], lw=1.2, marker=".", ms=3, label="dlog_reg")

    # Zero line for centered mode
    if centered:
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax.set_ylabel("Amplitude (baseline = 0)")
    else:
        ax.set_ylabel("Amplitude [0, 1]")

    ax.set_xlabel("Time (s)")
    title_suffix = " (HRF centered)" if centered else ""
    ax.set_title(f"{ch_name}{title_suffix}")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_regressor_overview")


# ═══════════════════════════════════════════════════════════════
#  Multi-panel regressor detail
# ═══════════════════════════════════════════════════════════════

def plot_regressor_panels(
    reginfo: dict,
    ch_name: str,
    fig_dir: Path | None,
    envelope: np.ndarray | None = None,
    srate_envelope: float | None = None,
    centered: bool = True,
) -> None:
    """Multi-panel regressor plot (one row per family).

    Parameters
    ----------
    centered : bool — if *True*, adds zero lines and adapts labels.
    """
    t_conv = reginfo["time_conv"]
    t_reg = reginfo["time_reg"]
    stride = max(1, len(t_conv) // 30000)
    sc = slice(None, None, stride)

    has_env = envelope is not None and srate_envelope is not None
    n_rows = 5 if has_env else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 3.2 * n_rows),
                             sharex=True)
    row = 0

    if has_env:
        t_env = np.arange(len(envelope)) / srate_envelope
        stride_e = max(1, len(t_env) // 30000)
        axes[row].plot(t_env[::stride_e], envelope[::stride_e],
                       ls="-", color="black", lw=0.6, alpha=0.7)
        axes[row].set_ylabel("Amplitude [0, 1]")
        axes[row].set_title(f"EMG envelope (corrected input) — {ch_name}")
        axes[row].grid(True, alpha=0.25)
        row += 1

    pairs = [
        ("conv",      "reg",      _COLORS[1], "HRF convolution"),
        ("dconv",     "dreg",     _COLORS[2], "Derivative of convolution"),
        ("log_conv",  "log_reg",  _COLORS[3], "Log HRF convolution"),
        ("dlog_conv", "dlog_reg", _COLORS[4], "Derivative of log convolution"),
    ]

    for conv_key, reg_key, color, title in pairs:
        ax = axes[row]
        ax.plot(t_conv[sc], reginfo[conv_key][sc],
                ls="-", color=color, lw=0.7, alpha=0.8, label=conv_key)
        ax.plot(t_reg, reginfo[reg_key],
                ls="-.", color=color, lw=1.3, marker=".", ms=4, label=reg_key)

        if centered:
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=0)
            ax.set_ylabel("Amplitude (baseline = 0)")
        else:
            ax.set_ylabel("Amplitude [0, 1]")

        # Show min/max annotation for centered mode
        if centered:
            vals = reginfo[conv_key]
            ax.text(
                0.01, 0.05,
                f"range: [{vals.min():.3f}, {vals.max():.3f}]",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

        ax.set_title(f"{title} — {ch_name}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.25)
        row += 1

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_regressor_panels")