"""Evaluate residual lines on an existing FARM output, without rerunning FARM.

The input BrainVision and its *_denoise.json sidecar are required. A distinct
*_lines output preserves the baseline. Reports compare the extra stage alone;
they measure spectral rejection and signal change, not physiological truth.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import mne
import numpy as np

from farm.config import FARMConfig
from farm import __version__
from farm.diagnostics.metrics import (
    compute_gradient_harmonic_reduction, compute_rms_reduction,
    compute_scanner_locked_reduction,
)
from farm.export.brainvision import export_brainvision
from farm.export.metadata import export_metadata
from farm.preprocessing.lines import clean_spectral_lines, line_cleanup_options
from farm.visualization.lines import plot_line_cleanup
from farm.visualization.spectra import plot_fft_power
from farm.preprocessing.filters import apply_bandpass


def evaluate(source: Path, output_dir: Path, *, mode="detect", figures=True,
             min_prominence_db=8., min_persistence=.75, max_lines=12, reference_path=None):
    source = source.resolve()
    metadata_path = source.with_name(f"{source.stem}_denoise.json")
    metadata = json.loads(metadata_path.read_text())
    if any(c.get("applied") for c in metadata.get("line_cleanup", {}).get("channels", [])):
        raise ValueError("Use the original FARM output, before line correction")
    names = metadata["selected_channels"]
    start, stop = metadata["scan_start_sample"], metadata["scan_stop_sample_exclusive"]
    config_fields = {f.name for f in fields(FARMConfig)}
    cfg = FARMConfig(**{k: v for k, v in metadata["config"].items() if k in config_fields})
    cfg.line_cleanup = mode
    cfg.line_min_prominence_db = min_prominence_db
    cfg.line_min_persistence = min_persistence
    cfg.line_max_count = max_lines
    cfg.validate()
    baseline = mne.io.read_raw_brainvision(str(source), preload=False, verbose=False)
    reference_path = Path(reference_path or metadata["input_vhdr"]).resolve()
    reference = mne.io.read_raw_brainvision(str(reference_path), preload=False, verbose=False)
    if (baseline.n_times != reference.n_times or baseline.ch_names != reference.ch_names or
            baseline.info["sfreq"] != reference.info["sfreq"]):
        raise ValueError("Original reference must match the full FARM recording")
    fs = float(baseline.info["sfreq"])
    cfg.validate_sampling_rate(fs)
    if not 0 <= start < stop <= baseline.n_times:
        raise ValueError("Invalid processed crop in FARM sidecar")
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{source.stem}_lines"
    full = baseline.get_data(picks=names).astype(np.float32)
    raw = reference.get_data(picks=names, start=start, stop=stop).astype(np.float32)
    reports = []
    for i, name in enumerate(names):
        before = full[i, start:stop].copy()
        after, report = clean_spectral_lines(
            before, fs, reference=raw[i], witness=raw[(i+1) % len(names)] if len(names)>1 else None,
            sdur=metadata["sdur_seconds"],
            tr=metadata["observed_tr_seconds"], **line_cleanup_options(cfg),
        )
        full[i, start:stop] = after
        report["channel"] = name
        report["witness_channel"] = names[(i+1) % len(names)] if len(names)>1 else None
        reports.append(report)
        if figures:
            plot_line_cleanup(before, after, fs, name, report, output_dir/"figures")
            plot_fft_power(apply_bandpass(after, fs, cfg.qc_bandpass), fs, name, output_dir/"figures")
        metrics = report.get("metrics", {})
        print(f"{name}: {report['applied_line_count']} lines corrected ({mode}), "
              f"band-limited envelope change={metrics.get('band_relative_envelope_change', 0):.4%}", flush=True)
    # Preserve explicit destructive masks in legacy inputs. Their indices are
    # absolute native samples and must remain marked as missing measurements.
    for a, b in metadata.get("masked_intervals_samples_full", []):
        full[:, a:b] = 0
    output = export_brainvision(baseline, full, [baseline.ch_names.index(n) for n in names],
                               str(output_dir), basename)
    after = full[:, start:stop]
    metadata.update(
        output_vhdr=output, software_version=__version__,
        line_cleanup_input_vhdr=str(source),
        line_cleanup={"mode": mode, "channels": reports},
        rms_reduction=compute_rms_reduction(raw, after, fs, names, cfg.qc_bandpass),
        scanner_locked_reduction=compute_scanner_locked_reduction(
            raw, after, fs, names, np.asarray(metadata["volume_onsets_samples_crop"]),
            cfg.tr, cfg.qc_bandpass),
        gradient_harmonic_reduction=compute_gradient_harmonic_reduction(
            raw, after, fs, names, metadata["sdur_seconds"], cfg.qc_bandpass),
    )
    metadata["config"].update({k: v for k, v in cfg.__dict__.items() if k.startswith("line_")})
    metadata["config"]["output_dir"] = str(output_dir)
    report_path = export_metadata(str(output_dir), basename, metadata)
    return output, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vhdr", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("detect", "notch", "regress"), default="detect")
    parser.add_argument("--min-prominence-db", type=float, default=8.)
    parser.add_argument("--min-persistence", type=float, default=.75)
    parser.add_argument("--max-lines", type=int, default=12)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    output, report = evaluate(args.vhdr, args.output_dir, mode=args.mode,
                              min_prominence_db=args.min_prominence_db,
                              min_persistence=args.min_persistence, max_lines=args.max_lines,
                              reference_path=args.reference, figures=not args.no_figures)
    print(f"BrainVision: {output}\nReport: {report}")


if __name__ == "__main__":
    main()
