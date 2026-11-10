"""Diagnostic comparisons for optional narrow spectral-line correction."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from .plotting import savefig


def plot_line_cleanup(before, after, fs, name, report, fig_dir: Path | None):
    length = report.get("window_samples", min(len(before), int(round(8*fs))))
    f, pb = sig.welch(before, fs=fs, nperseg=length)
    _, pa = sig.welch(after, fs=fs, nperseg=length)
    low, high = report["parameters"]["band_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(f, 10*np.log10(pb+1e-30), lw=.8, label="FARM")
    axes[0].plot(f, 10*np.log10(pa+1e-30), lw=.8, label="FARM + line stage")
    for candidate in report["candidates"]:
        color = "tab:red" if candidate["applied"] else "gray"
        axes[0].axvline(candidate["frequency_hz"], color=color, lw=.7, alpha=.5)
    axes[0].set_ylabel("PSD (dB V²/Hz)")
    axes[0].legend()
    axes[0].set_title(f"{name}: {report['applied_line_count']} lines corrected; gray = reported only")
    axes[1].plot(f, 10*np.log10((pa+1e-30)/(pb+1e-30)), lw=.8)
    axes[1].axhline(0, color="black", lw=.5)
    axes[1].set_ylabel("After / FARM (dB)")
    axes[1].set_xlabel("Hz")
    axes[1].set_xlim(low, high)
    fig.tight_layout()
    return savefig(fig, fig_dir, f"{name}_line_cleanup")
