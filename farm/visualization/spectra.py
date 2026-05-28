"""Frequency-domain visualisations: PSD, FFT power, spectrograms.

All functions write to ``fig_dir`` and never call ``plt.show()``.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from .plotting import savefig


def plot_psd_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
    title_before: str = "Before",
    title_after: str = "After",
    xlim: float = 375.0,
) -> None:
    """Side-by-side PSD (Welch) of two signals on the same channel."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, label, d, color in zip(
        axes,
        [title_before, title_after],
        [data_before, data_after],
        ["#d62728", "#2ca02c"],
    ):
        f_psd, psd = sig.welch(d, srate, nperseg=min(4096, len(d) // 4))
        ax.semilogy(f_psd, psd, color=color, linewidth=0.5)
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD")
        ax.set_title(f"{label} — {ch_name}")
        ax.set_xlim([0, xlim])
        ax.axvline(30, color="gray", ls="--", alpha=0.5, label="30 Hz")
        ax.legend()
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_psd_comparison")


def plot_fft_power(
    data_clean: np.ndarray,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
    xlim: float = 375.0,
) -> None:
    """FFT power (linear + dB) of the cleaned signal."""
    N = len(data_clean)
    freqs = np.fft.rfftfreq(N, d=1.0 / srate)
    fft_mag = np.abs(np.fft.rfft(data_clean.astype(np.float64))) / N
    power = fft_mag ** 2
    mask = freqs <= xlim

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].plot(freqs[mask], power[mask], linewidth=0.3, color="#2ca02c")
    axes[0].set_xlabel("Hz"); axes[0].set_ylabel("Power")
    axes[0].set_title(f"FFT Power (linear) — {ch_name}")
    axes[0].set_xlim([0, xlim])

    axes[1].plot(freqs[mask], 10 * np.log10(power[mask] + 1e-20),
                 linewidth=0.3, color="#2ca02c")
    axes[1].set_xlabel("Hz"); axes[1].set_ylabel("Power (dB)")
    axes[1].set_title(f"FFT Power (dB) — {ch_name}")
    axes[1].set_xlim([0, xlim])
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_fft_power")


def plot_fft_before_after(
    ts_before: np.ndarray,
    ts_after: np.ndarray,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
    sdur: float = 0.0,
    xlim: float = 375.0,
) -> None:
    """FFT dB plots: separate panels then overlay."""
    N = len(ts_before)
    freqs = np.fft.rfftfreq(N, d=1.0 / srate)
    db_b = 20 * np.log10(
        np.abs(np.fft.rfft(ts_before.astype(np.float64))) / N + 1e-20)
    db_a = 20 * np.log10(
        np.abs(np.fft.rfft(ts_after.astype(np.float64))) / N + 1e-20)

    # ── Separate panels ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for ax, label, db, color in zip(
        axes,
        [f"BEFORE — {ch_name}", f"AFTER FARM — {ch_name}"],
        [db_b, db_a],
        ["#d62728", "#2ca02c"],
    ):
        ax.plot(freqs, db, color=color, linewidth=0.3, alpha=0.8)
        ax.set_ylabel("dB"); ax.set_title(label)
        if sdur > 0:
            f_slice = 1.0 / sdur
            for h in range(1, 60):
                fh = f_slice * h
                if fh > xlim:
                    break
                ax.axvline(fh, color="orange", alpha=0.2, lw=0.5)
        ax.set_xlim([0, xlim])
    axes[1].set_xlabel("Hz")
    fig.suptitle(f"FFT before/after — {ch_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_fft_before_after")

    # ── Overlay ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(freqs, db_b, color="#d62728", lw=0.3, alpha=0.5, label="Before")
    ax.plot(freqs, db_a, color="#2ca02c", lw=0.3, alpha=0.8, label="After FARM")
    ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title(f"FFT overlay — {ch_name}"); ax.legend()
    ax.set_xlim([0, xlim])
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_fft_overlay")


def plot_psd_welch_before_after(
    ts_before: np.ndarray,
    ts_after: np.ndarray,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
    xlim: float = 375.0,
) -> None:
    """Welch PSD overlay before/after."""
    f_w, p_b = sig.welch(ts_before, srate,
                          nperseg=min(4096, len(ts_before) // 4))
    _, p_a = sig.welch(ts_after, srate,
                        nperseg=min(4096, len(ts_after) // 4))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.semilogy(f_w, p_b, alpha=0.6, label="Before")
    ax.semilogy(f_w, p_a, alpha=0.8, label="After FARM")
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD")
    ax.set_title(f"PSD Welch before/after — {ch_name}"); ax.legend()
    ax.set_xlim([0, xlim])
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_psd_welch_before_after")


def plot_spectrogram_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_name: str,
    fig_dir: Path | None,
    ylim: float = 500.0,
) -> None:
    """Side-by-side spectrograms before/after."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, label, ts in zip(
        axes, ["Before (HPF only)", "After FARM"],
        [data_before, data_after],
    ):
        nperseg_s = min(512, int(srate * 0.25))
        f_sp, t_sp, Sxx = sig.spectrogram(
            ts, fs=srate, nperseg=nperseg_s, noverlap=nperseg_s // 2)
        im = ax.pcolormesh(
            t_sp, f_sp, 10 * np.log10(Sxx + 1e-20),
            shading="gouraud", cmap="inferno")
        ax.set_ylabel("Hz"); ax.set_title(f"{label} — {ch_name}")
        ax.set_ylim([0, ylim])
        plt.colorbar(im, ax=ax, label="dB")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, fig_dir, f"{ch_name}_spectrogram_comparison")