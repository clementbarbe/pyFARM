"""Regressor diagnostic plots.

- ``plot_envelope_correction`` — baseline correction diagnostic.
- ``plot_regressor`` — single-axis overview (MATLAB farm_plot_regressor style).
- ``plot_regressor_panels`` — multi-panel showing ALL 8 regressors + derivatives.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plotting import savefig


_COLORS = {
    "conv":      "#D95319",  # orange
    "dconv":     "#EDB120",  # yellow
    "log_conv":  "#7E2F8E",  # purple
    "dlog_conv": "#77AC30",  # green
    "mod":       "#0072BD",  # blue
    "log_mod":   "#A2142F",  # dark red
    "dmod":      "#4DBEEE",  # cyan
    "dlog_mod":  "#D95319",  # orange (reused, different panel)
}


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

    ax = axes[0]
    ax.plot(t[sl], envelope_raw[sl], color="black", lw=0.4, alpha=0.7,
            label="Original envelope")
    if baseline is not None:
        ax.plot(t[sl], baseline[sl], color="#d62728", lw=1.5, alpha=0.9,
                label="Estimated baseline")
        ax.fill_between(t[sl], 0, baseline[sl], color="#d62728", alpha=0.08)
    if threshold_value > 0 and baseline is not None:
        thresh_line = np.clip(baseline + threshold_value, 0, 1)
        ax.plot(t[sl], thresh_line[sl], color="#FF9800", lw=1.0, ls="--",
                alpha=0.8, label=f"Threshold (+{threshold_value:.4f})")
    ax.set_ylabel("Amplitude [0, 1]")
    ax.set_title(f"Envelope baseline correction — {ch_name} — {method}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(t[sl], envelope_raw[sl], color="#d62728", lw=0.3, alpha=0.3,
            label="Before")
    ax.plot(t[sl], envelope_corrected[sl], color="#2ca02c", lw=0.5, alpha=0.9,
            label="After")
    ax.set_ylabel("Amplitude [0, 1]")
    ax.set_title(f"Corrected envelope — {ch_name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.2)
    pct_zero = 100.0 * np.mean(envelope_corrected == 0)
    ax.text(0.01, 0.95,
            f"Zeroed: {pct_zero:.1f}%  |  σ_noise: {noise_std:.4f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax = axes[2]
    zoom_half = int(min(10.0 * srate, N // 4))
    zoom_center = _find_transition(envelope_corrected, srate)
    z_start = max(0, zoom_center - zoom_half)
    z_stop = min(N, zoom_center + zoom_half)
    ax.plot(t[z_start:z_stop], envelope_raw[z_start:z_stop],
            color="#d62728", lw=0.6, alpha=0.5, label="Before")
    ax.plot(t[z_start:z_stop], envelope_corrected[z_start:z_stop],
            color="#2ca02c", lw=1.0, alpha=0.9, label="After")
    if baseline is not None:
        ax.plot(t[z_start:z_stop], baseline[z_start:z_stop],
                color="#d62728", lw=1.2, ls="--", alpha=0.6, label="Baseline")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Zoom quiet↔active — {ch_name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_envelope_correction")


def _find_transition(envelope, srate, min_jump=0.2):
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
#  Single-axis regressor overview (MATLAB style)
# ═══════════════════════════════════════════════════════════════

def plot_regressor(
    reginfo: dict,
    ch_name: str,
    fig_dir: Path | None,
    envelope: np.ndarray | None = None,
    srate_envelope: float | None = None,
    centered: bool = True,
) -> None:
    """Single-axis overview of all regressor traces."""
    t_conv = reginfo["time_conv"]
    t_reg = reginfo["time_reg"]
    stride = max(1, len(t_conv) // 30000)
    sc = slice(None, None, stride)

    fig, ax = plt.subplots(figsize=(18, 6))

    if envelope is not None and srate_envelope is not None:
        t_env = np.arange(len(envelope)) / srate_envelope
        stride_e = max(1, len(t_env) // 30000)
        ax.plot(t_env[::stride_e], envelope[::stride_e],
                ls="-", color="black", lw=0.6, alpha=0.4, label="envelope")

    # Convolved (hires + TR)
    for conv_k, reg_k, color, label in [
        ("conv",      "reg",      _COLORS["conv"],      "conv"),
        ("dconv",     "dreg",     _COLORS["dconv"],     "dconv"),
        ("log_conv",  "log_reg",  _COLORS["log_conv"],  "log_conv"),
        ("dlog_conv", "dlog_reg", _COLORS["dlog_conv"], "dlog_conv"),
    ]:
        ax.plot(t_conv[sc], reginfo[conv_k][sc],
                ls="-", color=color, lw=0.6, alpha=0.7, label=label)
        ax.plot(t_reg, reginfo[reg_k],
                ls="-.", color=color, lw=1.0, marker=".", ms=2,
                label=f"{reg_k} (TR)")

    if centered:
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax.set_ylabel("Amplitude (baseline = 0)")
    else:
        ax.set_ylabel("Amplitude [0, 1]")

    ax.set_xlabel("Time (s)")
    ax.set_title(f"Regressor overview — {ch_name}")
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_regressor_overview")


# ═══════════════════════════════════════════════════════════════
#  Multi-panel: ALL 8 regressors + derivatives
# ═══════════════════════════════════════════════════════════════

def plot_regressor_panels(
    reginfo: dict,
    ch_name: str,
    fig_dir: Path | None,
    envelope: np.ndarray | None = None,
    srate_envelope: float | None = None,
    centered: bool = True,
) -> None:
    """Multi-panel plot showing ALL produced regressors.

    8 panels (+ 1 for envelope if provided):

    1. Envelope (input)
    2. conv + reg — HRF convolution
    3. dconv + dreg — Derivative of HRF convolution
    4. log_conv + log_reg — Log-compressed HRF convolution
    5. dlog_conv + dlog_reg — Derivative of log-compressed
    6. mod — Direct modulation (TR only)
    7. dmod — Derivative of direct modulation (TR only)
    8. log_mod — Log-compressed modulation (TR only)
    9. dlog_mod — Derivative of log-compressed modulation (TR only)
    """
    t_conv = reginfo["time_conv"]
    t_reg = reginfo["time_reg"]
    stride = max(1, len(t_conv) // 30000)
    sc = slice(None, None, stride)

    has_env = envelope is not None and srate_envelope is not None

    # Build panel definitions
    panels = []

    if has_env:
        panels.append({
            "type": "envelope",
            "title": f"EMG envelope (corrected input) — {ch_name}",
        })

    # Convolution-based panels (hires + TR)
    hrf_panels = [
        ("conv",      "reg",      _COLORS["conv"],
         "HRF Convolution", True),
        ("dconv",     "dreg",     _COLORS["dconv"],
         "Derivative of HRF Convolution", True),
        ("log_conv",  "log_reg",  _COLORS["log_conv"],
         "Log-compressed HRF Convolution", True),
        ("dlog_conv", "dlog_reg", _COLORS["dlog_conv"],
         "Derivative of Log-compressed HRF Convolution", True),
    ]
    for conv_k, reg_k, color, title, use_centered in hrf_panels:
        panels.append({
            "type": "hires",
            "conv_key": conv_k,
            "reg_key": reg_k,
            "color": color,
            "title": f"{title} — {ch_name}",
            "centered": use_centered,
        })

    # Modulation panels (TR only, always [0, 1])
    mod_panels = [
        ("mod",      _COLORS["mod"],      "Direct Modulation (non-convolved)"),
        ("dmod",     _COLORS["dmod"],     "Derivative of Direct Modulation"),
        ("log_mod",  _COLORS["log_mod"],  "Log-compressed Modulation (non-convolved)"),
        ("dlog_mod", _COLORS["dlog_mod"], "Derivative of Log-compressed Modulation"),
    ]
    for key, color, title in mod_panels:
        panels.append({
            "type": "tr_only",
            "key": key,
            "color": color,
            "title": f"{title} — {ch_name}",
        })

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.8 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, panel in enumerate(panels):
        ax = axes[row]

        if panel["type"] == "envelope":
            t_env = np.arange(len(envelope)) / srate_envelope
            stride_e = max(1, len(t_env) // 30000)
            ax.plot(t_env[::stride_e], envelope[::stride_e],
                    ls="-", color="black", lw=0.6, alpha=0.7)
            ax.set_ylabel("[0, 1]")
            ax.set_ylim([-0.05, 1.1])

        elif panel["type"] == "hires":
            color = panel["color"]
            conv_k = panel["conv_key"]
            reg_k = panel["reg_key"]

            ax.plot(t_conv[sc], reginfo[conv_k][sc],
                    ls="-", color=color, lw=0.7, alpha=0.8,
                    label=f"{conv_k} (hires)")
            ax.plot(t_reg, reginfo[reg_k],
                    ls="none", color=color, marker="o", ms=3, alpha=0.9,
                    label=f"{reg_k} (TR)")

            if centered and panel["centered"]:
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5,
                           zorder=0)
                ax.set_ylabel("baseline = 0")
                vals = reginfo[conv_k]
                ax.text(
                    0.01, 0.05,
                    f"range: [{vals.min():.3f}, {vals.max():.3f}]",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            else:
                ax.set_ylabel("[0, 1]")

            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        elif panel["type"] == "tr_only":
            color = panel["color"]
            key = panel["key"]
            ax.plot(t_reg, reginfo[key],
                    ls="-", color=color, lw=1.2, marker="o", ms=3,
                    alpha=0.9, label=f"{key} (TR)")
            ax.set_ylabel("[0, 1]")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        ax.set_title(panel["title"], fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_regressor_panels")