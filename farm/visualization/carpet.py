"""Carpet (image) plots of stacked slice segments."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plotting import savefig


def plot_carpet(
    segments: np.ndarray,
    fig_dir: Path | None,
    name: str = "carpet",
    title: str = "Carpet",
    xlabel: str = "Sample",
    ylabel: str = "Segment #",
) -> None:
    """Display a carpet plot from a 2-D segment array."""
    if segments.size == 0:
        return
    vmax = np.percentile(np.abs(segments), 99)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(segments, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    savefig(fig, fig_dir, name)


def plot_carpet_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    srate: float,
    ch_name: str,
    vol_onsets: np.ndarray,
    sdur: float,
    n_sg: int,
    n_vol: int,
    fig_dir: Path | None,
    bandpass: tuple | None = None,
) -> None:
    """Before/after carpet plots using band-passed data."""
    from farm.preprocessing.filters import apply_bandpass

    seg_len = int(round(sdur * srate))
    for stage, arr, suffix in [
        ("BEFORE (HPF)", data_before, "before"),
        ("AFTER FARM", data_after, "after"),
    ]:
        ts = apply_bandpass(arr, srate, bandpass) if bandpass else arr
        rows = []
        for v in range(n_vol):
            for k in range(n_sg):
                s = int(vol_onsets[v] + round(k * sdur * srate))
                e = s + seg_len
                if e <= len(ts):
                    rows.append(ts[s:e])
        if rows:
            plot_carpet(
                np.array(rows),
                fig_dir,
                name=f"{ch_name}_carpet_{suffix}",
                title=f"Carpet — {ch_name} — {stage}",
            )