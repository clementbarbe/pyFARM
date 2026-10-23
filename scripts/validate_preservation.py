"""Measure recovery of independent EMG injected into a real scan excerpt.

Run from the repository after installing it. This does not export or modify the
recording. Recovery is D(raw + injection) - D(raw), including template/PCA
adaptation to the injection, at verified fixed protocol timing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal as sig

from farm.config import FARMConfig
from farm.export.metadata import _jsonable
from farm.diagnostics.metrics import compute_scanner_locked_reduction, compute_gradient_harmonic_reduction
from farm.io.loader import load_brainvision, select_channels
from farm.io.triggers import detect_volume_onsets
from farm.preprocessing.filters import hpf_fir
from farm.preprocessing.resampling import upsample, downsample
from farm.preprocessing.trim import trim_to_scan
from farm.timing.optimization import compute_slice_markers
from farm.utils.slices import build_slice_info
from farm.workflow import _correct_channel


def validate(cfg: FARMConfig, rms_uv: float = 20.0, seed: int = 201) -> dict:
    """Return amplitude, waveform and 50-ms RMS-envelope recovery per channel."""
    cfg.validate()
    if rms_uv <= 0 or cfg.timing_mode != "fixed":
        raise ValueError("Use positive injection RMS and verified fixed timing")
    raw = load_brainvision(cfg.vhdr_path)
    fs = float(raw.info["sfreq"])
    cfg.validate_sampling_rate(fs)
    volumes, _, _ = detect_volume_onsets(
        raw, cfg.trigger, cfg.tr, cfg.n_volumes,
        max_interval_error_seconds=cfg.trigger_tolerance_seconds,
    )
    data, names, _ = select_channels(raw, cfg.ch_regex)
    data, volumes, _, _, _, count = trim_to_scan(data, volumes, fs, cfg.tr)
    sdur = cfg.fixed_sdur_seconds
    gap = float(np.median(np.diff(volumes))) / fs - cfg.n_sg * sdur
    if gap < 0:
        raise ValueError("Fixed acquisition duration exceeds measured TR")
    t = np.arange(data.shape[1]) / fs
    envelope = np.full(len(t), 0.15)
    for center in [2.31, 8.57, 16.81, 27.23, 34.17]:
        envelope += np.exp(-0.5*((t-center)/0.13)**2)
    sos = sig.butter(4, [45, 220], btype="bandpass", fs=fs, output="sos")
    kernel = np.ones(max(1, int(round(.05*fs)))) / max(1, int(round(.05*fs)))
    results = []
    for channel, name in enumerate(names):
        rng = np.random.default_rng(seed + channel)
        injected = (sig.sosfiltfilt(sos, rng.standard_normal(len(t))) * envelope).astype(np.float32)
        injected *= rms_uv * 1e-6 / np.sqrt(np.mean(injected**2))
        outputs = []
        for original in (data[channel], data[channel] + injected):
            data_up, fs_up, vol_up = upsample(original[None, :], fs, volumes, cfg.interp_factor)
            reference = hpf_fir(original[None, :], fs, cfg.artifact_hpf_cutoff)
            ref_up, _, _ = upsample(reference, fs, volumes, cfg.interp_factor)
            markers, errors, length = compute_slice_markers(vol_up, sdur, gap, fs_up, cfg.n_sg)
            cleaned = _correct_channel(
                data_up[0], ref_up[0], markers, errors, length,
                build_slice_info(len(markers), cfg.n_sg, cfg.window_size),
                0, len(data_up[0]), int(round(gap*fs_up)), cfg, fs_up,
            )
            correction = downsample((data_up[0]-cleaned)[None, :], fs_up,
                                    cfg.interp_factor, len(t))[0]
            outputs.append(original - correction)
        recovered = outputs[1] - outputs[0]
        expected_env = np.sqrt(np.convolve(injected.astype(np.float64)**2, kernel, mode="same"))
        recovered_env = np.sqrt(np.convolve(recovered.astype(np.float64)**2, kernel, mode="same"))
        results.append({
            "channel": name,
            "gain": float(np.dot(recovered, injected) / np.dot(injected, injected)),
            "relative_waveform_error": float(np.linalg.norm(recovered-injected) / np.linalg.norm(injected)),
            "correlation": float(np.corrcoef(recovered, injected)[0, 1]),
            "relative_envelope_error": float(np.linalg.norm(recovered_env-expected_env) / np.linalg.norm(expected_env)),
            "scanner_locked": compute_scanner_locked_reduction(
                data[channel][None, :], outputs[0][None, :], fs, [name], volumes,
                cfg.tr, cfg.qc_bandpass,
            )[0],
            "gradient_harmonics": compute_gradient_harmonic_reduction(
                data[channel][None, :], outputs[0][None, :], fs, [name], sdur,
                cfg.qc_bandpass,
            )[0],
        })
    return {"input_vhdr": str(Path(cfg.vhdr_path).resolve()), "n_volumes": count,
            "sampling_rate_hz": fs, "injection_band_hz": [45, 220],
            "injection_rms_uv": rms_uv, "seed": seed,
            "config": dict(cfg.__dict__), "channels": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vhdr")
    parser.add_argument("--tr", type=float, required=True)
    parser.add_argument("--n-slices", type=int, required=True)
    parser.add_argument("--mb-factor", type=int, default=1)
    parser.add_argument("--sdur-ms", type=float, required=True)
    parser.add_argument("--n-volumes", type=int, default=24)
    parser.add_argument("--rms-uv", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--pca-max-components", type=int, default=4)
    parser.add_argument("--trigger", default="R128")
    parser.add_argument("--ch-regex", default="EXT|FLE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = FARMConfig(vhdr_path=args.vhdr, tr=args.tr, n_slices=args.n_slices,
                     mb_factor=args.mb_factor, timing_mode="fixed",
                     fixed_sdur_seconds=args.sdur_ms/1000, n_volumes=args.n_volumes,
                     pca_max_components=args.pca_max_components, figures_dir="",
                     trigger=args.trigger, ch_regex=args.ch_regex)
    result = validate(cfg, args.rms_uv, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    for row in result["channels"]:
        print(f"{row['channel']}: gain={row['gain']:.3f}, "
              f"waveform error={row['relative_waveform_error']:.3f}, "
              f"envelope error={row['relative_envelope_error']:.3f}")


if __name__ == "__main__":
    main()
