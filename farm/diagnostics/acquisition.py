"""Acquisition-limit diagnostics; never infer recovery from denoising success."""

from pathlib import Path
import re

import numpy as np


def assess_adc_rails(data, fs, channel_info, names, volume_onsets, source_vhdr):
    """Count samples at signed 16-bit rails using calibrated SI amplitudes.

    BrainAmp recordings can use symmetric +/-32767 limits, so checking only
    -32768 would miss negative clipping. A rail hit is a QC observation, not a
    measurement of the unknown pre-clipping waveform. Other formats are marked
    unassessed rather than declared free of saturation.
    """
    header = Path(source_vhdr).read_text(encoding="utf-8-sig", errors="replace")
    supported = bool(re.search(r"^BinaryFormat\s*=\s*INT_16\s*$", header, re.M | re.I))
    results = []
    onsets = np.asarray(volume_onsets, dtype=np.int64)
    for x, info, name in zip(data, channel_info, names):
        report = {"channel": name, "assessed": supported}
        if not supported:
            report["reason"] = "ADC limits unknown for this storage format"
            results.append(report)
            continue
        scale = float(info["cal"])*float(info["range"])
        if not np.isfinite(scale) or scale <= 0:
            report.update(assessed=False, reason="Invalid channel calibration")
            results.append(report)
            continue
        rails = (x <= -32766.5*scale) | (x >= 32766.5*scale)
        changes = np.flatnonzero(np.diff(np.r_[False, rails, False].astype(np.int8)))
        longest = int(np.max(changes[1::2]-changes[::2])) if len(changes) else 0
        counts = [int(np.count_nonzero(rails[start:stop]))
                  for start, stop in zip(onsets, np.r_[onsets[1:], len(x)])]
        report.update(rail_samples=int(rails.sum()), rail_fraction=float(rails.mean()),
                      longest_rail_seconds=longest/fs,
                      affected_volume_indices=[i for i, n in enumerate(counts) if n],
                      rail_samples_by_volume=counts,
                      rail_amplitude_volts=32767*scale)
        results.append(report)
    return results
