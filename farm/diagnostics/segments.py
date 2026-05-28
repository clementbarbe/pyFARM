"""Per-step segment diagnostics (carpet + mean ± std + histogram)."""

import numpy as np
import matplotlib.pyplot as plt


def diagnose_segments(
    ch_data: np.ndarray,
    onsets: np.ndarray,
    seg_len: int,
    ch_name: str,
    stage: str,
    n_show: int = 200,
) -> None:
    """Visualise the first *n_show* segments at a given pipeline stage.

    Displays: carpet plot, mean ± std, amplitude histogram.
    """
    rows = []
    for idx in range(min(n_show, len(onsets))):
        start = int(onsets[idx])
        stop = start + seg_len
        if 0 <= start and stop <= len(ch_data):
            rows.append(ch_data[start:stop])
    if not rows:
        return

    carpet = np.asarray(rows, dtype=np.float32)
    vmax = np.percentile(np.abs(carpet), 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].imshow(carpet, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Carpet — {ch_name} — {stage}")
    axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Segment #")

    mean_seg = carpet.mean(axis=0)
    std_seg = carpet.std(axis=0)
    axes[1].plot(mean_seg, linewidth=0.6, label="Mean")
    axes[1].fill_between(
        np.arange(len(mean_seg)),
        mean_seg - std_seg, mean_seg + std_seg,
        alpha=0.3, label="±1 std",
    )
    axes[1].set_title("Mean segment ± std"); axes[1].legend(fontsize=8)

    axes[2].hist(carpet.ravel(), bins=100, color="steelblue", alpha=0.7)
    axes[2].set_title("Amplitude distribution"); axes[2].set_xlabel("Amplitude")

    fig.tight_layout()
    plt.show()