"""Export cleaned data as BrainVision triplet (.vhdr/.eeg/.vmrk)."""

import logging
from pathlib import Path

import numpy as np
import mne
from mne.io.constants import FIFF

logger = logging.getLogger("farm.export.brainvision")


def _annotation_events(raw: mne.io.Raw) -> list[dict]:
    """Convert annotations to integer samples by rounding, never truncation.

    A mathematically integral timestamp can multiply back to e.g. 930622.99999.
    Truncation in MNE's generic export path then moves its marker one sample.
    pybv accepts explicit sample positions, avoiding this timing distortion.
    """
    onsets = raw.time_as_index(raw.annotations.onset - raw.first_time,
                               use_rounding=True)
    events = []
    for onset, annotation in zip(onsets, raw.annotations):
        onset = int(onset)
        if not 0 <= onset < raw.n_times:
            raise ValueError("Annotation onset outside the recording cannot be exported")
        duration = min(int(round(annotation["duration"] * raw.info["sfreq"])),
                       raw.n_times - onset)
        description = annotation["description"]
        event_type = "Comment"
        for prefix, kind in [("Stimulus/S", "Stimulus"), ("Response/R", "Response")]:
            if description.startswith(prefix) and description[len(prefix):].strip().isdigit():
                description = int(description[len(prefix):].strip())
                event_type = kind
                break
        if isinstance(description, str) and description.startswith("Comment/"):
            description = description[len("Comment/"):]
        event = dict(onset=onset, duration=duration, description=description, type=event_type)
        if annotation.get("ch_names"):
            event["channels"] = list(annotation["ch_names"])
        events.append(event)
    return events


def export_brainvision(
    raw_original: mne.io.Raw,
    data_clean_full: np.ndarray,
    ch_indices: list[int],
    output_dir: str,
    basename: str,
) -> str:
    """Write a BrainVision file set with cleaned selected channels.

    All original channels are retained; only selected EMG channels are
    replaced.  Original annotations are preserved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vhdr_path = str(out / f"{basename}.vhdr")

    clean = np.asarray(data_clean_full)
    if clean.ndim != 2:
        raise ValueError("data_clean_full must have shape (n_channels, n_samples)")
    if clean.shape[0] != len(ch_indices):
        raise ValueError("data_clean_full channel count and ch_indices length do not match")
    if clean.shape[1] != raw_original.n_times:
        raise ValueError(
            f"Cleaned data has {clean.shape[1]} samples but Raw has {raw_original.n_times}."
        )

    raw_data = raw_original.get_data().copy()
    for local_index, raw_channel_index in enumerate(ch_indices):
        raw_data[raw_channel_index] = clean[local_index].astype(np.float64)

    try:
        from pybv import write_brainvision
    except ImportError as exc:
        raise RuntimeError(
            "BrainVision export requires 'pybv'. Install with `python -m pip install -e .`."
        ) from exc
    # Match the calibrated-unit policy of MNE's BrainVision exporter. Only
    # voltage channels are rescaled; other physical measurements remain in SI.
    units = {FIFF.FIFF_UNIT_V: "µV", FIFF.FIFF_UNIT_CEL: "°C",
             FIFF.FIFF_UNIT_T: "T", FIFF.FIFF_UNIT_T_M: "T/m",
             FIFF.FIFF_UNIT_MOL: "M", FIFF.FIFF_UNIT_S: "S",
             FIFF.FIFF_UNIT_SEC: "s"}
    if hasattr(FIFF, "FIFF_UNIT_PX"):
        units[FIFF.FIFF_UNIT_PX] = "px"
    write_brainvision(
        data=raw_data, sfreq=float(raw_original.info["sfreq"]),
        ch_names=raw_original.ch_names, fname_base=basename, folder_out=str(out),
        overwrite=True, events=_annotation_events(raw_original),
        resolution=0.1, unit=[units.get(ch["unit"], "n/a") for ch in raw_original.info["chs"]],
        fmt="binary_float32", meas_date=raw_original.info["meas_date"],
    )

    logger.info("BrainVision exported: %s", vhdr_path)
    return vhdr_path
