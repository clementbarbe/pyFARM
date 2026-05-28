"""BrainVision file loading and channel selection."""

import logging
import re
from pathlib import Path

import numpy as np
import mne

logger = logging.getLogger("farm.io.loader")


def load_brainvision(vhdr_path: str) -> mne.io.Raw:
    """Load a BrainVision ``.vhdr`` file and return the MNE Raw object.

    The data are preloaded into memory.
    """
    vhdr = Path(vhdr_path)
    assert vhdr.exists(), f"File not found: {vhdr}"
    assert vhdr.with_suffix(".eeg").exists(), ".eeg file missing"
    assert vhdr.with_suffix(".vmrk").exists(), ".vmrk file missing"

    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    logger.info(
        "Loaded %s — %d ch, %d samples, %.0f Hz, %.1f s",
        vhdr.name, len(raw.ch_names), raw.n_times,
        raw.info["sfreq"], raw.n_times / raw.info["sfreq"],
    )
    return raw


def select_channels(
    raw: mne.io.Raw, ch_regex: str
) -> tuple:
    """Select channels whose names match *ch_regex*.

    Parameters
    ----------
    raw : MNE Raw object.
    ch_regex : str — regular expression.

    Returns
    -------
    data : ndarray, shape ``(n_selected, n_samples)``, float32.
    ch_names : list of str.
    ch_indices : list of int — indices in the original Raw.
    """
    pattern = re.compile(ch_regex, re.IGNORECASE)
    ch_indices = [i for i, n in enumerate(raw.ch_names) if pattern.search(n)]
    if not ch_indices:
        raise ValueError(
            f"No channel matches '{ch_regex}'. Available: {raw.ch_names}"
        )
    ch_names = [raw.ch_names[i] for i in ch_indices]
    data = raw.get_data()[ch_indices].astype(np.float32)
    logger.info("Selected %d channels: %s", len(ch_names), ch_names)
    return data, ch_names, ch_indices