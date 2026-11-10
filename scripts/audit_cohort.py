"""Read FARM sidecars and source ADC samples; never modify cohort recordings."""

import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np

from farm.diagnostics.acquisition import assess_adc_rails


def audit(root, output_dir):
    paths = sorted(root.rglob("*_denoise.json"))
    if not paths:
        raise ValueError("No FARM sidecars found")
    rows, disagreements, errors = [], [], []
    for path in paths:
        try:
            m = json.loads(path.read_text())
            raw = mne.io.read_raw_brainvision(m["input_vhdr"], preload=False, verbose=False)
            names = m["selected_channels"]
            indices = [raw.ch_names.index(n) for n in names]
            data = raw.get_data(picks=indices, start=m["scan_start_sample"],
                                stop=m["scan_stop_sample_exclusive"]).astype(np.float32)
            fs = float(raw.info["sfreq"])
            rails = assess_adc_rails(data, fs, [raw.info["chs"][i] for i in indices],
                                     names, m["volume_onsets_samples_crop"], m["input_vhdr"])
            estimates = np.asarray(m["initial_sdur_by_channel_seconds"])
            if np.ptp(estimates) > 5/(fs*m["n_slice_groups"]):
                disagreements.append(dict(run=path.stem, estimates_ms=(estimates*1000).tolist(),
                                          used_duration_ms=m["sdur_seconds"]*1000))
            rms = {r["ch_name"]: r for r in m["rms_reduction"]}
            locked = {r["ch_name"]: r for r in m["scanner_locked_reduction"]}
            for r in rails:
                name = r["channel"]
                rows.append(dict(
                    run=path.stem.replace("_FARM_denoise", ""), channel=name,
                    adc_assessed=r["assessed"], rail_fraction=r.get("rail_fraction"),
                    rail_samples=r.get("rail_samples"), longest_rail_seconds=r.get("longest_rail_seconds"),
                    affected_volume_count=len(r.get("affected_volume_indices", [])),
                    sdur_ms=m["sdur_seconds"]*1000,
                    cleaned_rms_uv=rms[name]["rms_after"]*1e6,
                    residual_locked_rms_uv=locked[name]["locked_rms_after"]*1e6,
                    source_vhdr=m["input_vhdr"], sidecar=str(path),
                ))
        except Exception as exc:
            errors.append(dict(sidecar=str(path), error=f"{type(exc).__name__}: {exc}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir/"channel_audit.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        sidecars=len(paths), channels=len(rows), subjects=len({p.parent.name for p in paths}),
        runs_with_rail_samples=len({r["sidecar"] for r in rows if r["rail_samples"]}),
        runs_above_one_percent_rail_samples=len({r["sidecar"] for r in rows if (r["rail_fraction"] or 0) > .01}),
        channel_timing_disagreements=disagreements, read_errors=errors,
        note="Rail presence requires review; absence does not validate EMG recovery.",
    )
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
