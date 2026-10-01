"""Export cleaned data as BrainVision triplet (.vhdr/.eeg/.vmrk)."""

import logging
from pathlib import Path

import numpy as np
import mne

logger = logging.getLogger("farm.export.brainvision")


def export_brainvision(
    raw_original: mne.io.Raw,
    data_clean_full: np.ndarray,
    ch_indices: list[int],
    output_dir: str,
    basename: str,
) -> str:
    """Write a BrainVision file set with cleaned selected channels.

    All original channels are retained and only selected EMG channels are
    replaced. Original annotations are copied and first_samp is preserved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vhdr_path = str(out / f"{basename}.vhdr")

    clean = np.asarray(data_clean_full)

    if clean.ndim != 2:
        raise ValueError("data_clean_full must have shape (n_channels, n_samples)")

    if clean.shape[0] != len(ch_indices):
        raise ValueError(
            "data_clean_full channel count and ch_indices length do not match"
        )

    if clean.shape[1] != raw_original.n_times:
        raise ValueError(
            f"Cleaned data has {clean.shape[1]} samples but Raw has "
            f"{raw_original.n_times} samples."
        )

    raw_data = raw_original.get_data().copy()

    for local_index, raw_channel_index in enumerate(ch_indices):
        raw_data[raw_channel_index] = clean[local_index].astype(np.float64)

    raw_export = mne.io.RawArray(
        raw_data,
        raw_original.info.copy(),
        first_samp=raw_original.first_samp,
        verbose=False,
    )

    raw_export.set_annotations(raw_original.annotations.copy())

    try:
        mne.export.export_raw(
            vhdr_path,
            raw_export,
            overwrite=True,
            verbose=False,
        )
    except RuntimeError as exc:
        if "pybv" not in str(exc).lower():
            raise
        logger.warning(
            "BrainVision export skipped because optional dependency 'pybv' "
            "is not installed. NPZ/MAT exports will still be written."
        )
        return ""

    logger.info("BrainVision exported: %s", vhdr_path)
    return vhdr_path