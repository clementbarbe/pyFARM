"""Export cleaned data as BrainVision triplet (.vhdr/.eeg/.vmrk)."""

import logging
from pathlib import Path

import numpy as np
import mne

logger = logging.getLogger("farm.export.brainvision")


def export_brainvision(
    raw_original: mne.io.Raw,
    data_clean_full: np.ndarray,
    ch_indices: list,
    output_dir: str,
    basename: str,
) -> str:
    """Write a BrainVision file set with the cleaned channels.

    All channels are exported; only the channels listed in *ch_indices*
    are replaced with the cleaned data.  All original markers are preserved.

    Returns
    -------
    str — path to the written ``.vhdr`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vhdr_path = str(out / f"{basename}.vhdr")

    raw_export = raw_original.copy()
    raw_data = raw_export.get_data()
    for i, ch_i in enumerate(ch_indices):
        raw_data[ch_i] = data_clean_full[i].astype(np.float64)
    raw_export = mne.io.RawArray(raw_data, raw_original.info.copy(), verbose=False)
    raw_export.set_annotations(raw_original.annotations.copy())

    mne.export.export_raw(vhdr_path, raw_export, overwrite=True, verbose=False)
    logger.info("BrainVision exported: %s", vhdr_path)
    return vhdr_path