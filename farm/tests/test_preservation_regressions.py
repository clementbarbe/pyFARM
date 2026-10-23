"""Scientific regression tests with independently known EMG and artifacts."""

import json

import mne
import numpy as np
import pytest
from scipy import signal as sig

from farm.alignment.phase_shift import extract_aligned_segments, restore_artifact_segments
from farm.config import FARMConfig
from farm.export.brainvision import export_brainvision
from farm.export.metadata import export_metadata
from farm.diagnostics.metrics import compute_gradient_harmonic_reduction
from farm.io.triggers import detect_volume_onsets
from farm.pca.cleanup import pca_cleanup
from farm.preprocessing.filters import hpf_fir
from farm.preprocessing.resampling import upsample, downsample
from farm.templates.subtraction import build_artifact_templates
from farm.templates.zerofill import gap_mask_intervals
from farm.timing.estimation import estimate_initial_timing
from farm.timing.optimization import compute_slice_markers
from farm.utils.slices import build_slice_info
from farm.workflow import _correct_channel, run_pipeline


def synthetic_emg(seed=11, n_vol=32):
    fs, tr, sdur, n_sg = 2000.0, 0.46, 0.08137, 4
    onsets = np.rint(np.arange(n_vol) * tr * fs).astype(np.int64)
    t = np.arange(int(round(n_vol * tr * fs))) / fs
    rng = np.random.default_rng(seed)
    sos = sig.butter(4, [45, 320], btype="bandpass", fs=fs, output="sos")
    carrier = sig.sosfiltfilt(sos, rng.standard_normal(len(t)))
    envelope = 0.2 + sum(np.exp(-0.5 * ((t-c) / 0.12)**2) for c in [1.7, 5.8, 10.3])
    emg = (carrier * envelope + 0.05 * np.sin(2*np.pi*7*t)).astype(np.float32)
    artifact = np.zeros_like(t)
    for v, onset in enumerate(onsets / fs):
        for group in range(n_sg):
            tau = t - onset - group * sdur
            support = (tau >= 0) & (tau < sdur)
            artifact[support] += (20 + group) * (1 + 0.02*np.sin(v/5)) * (
                np.sin(2*np.pi*180*tau[support]) * np.exp(-((tau[support]-.028)/.016)**2)
            )
        tau = t - onset - n_sg*sdur
        support = (tau >= 0) & (tau < tr - n_sg*sdur)
        artifact[support] += 18 * np.sin(2*np.pi*125*tau[support]) * np.exp(-((tau[support]-.04)/.018)**2)
    cfg = FARMConfig(vhdr_path="synthetic.vhdr", tr=tr, n_slices=n_sg, mb_factor=1,
                     interp_factor=4, timing_mode="fixed", fixed_sdur_seconds=sdur,
                     figures_dir="")
    return emg, artifact.astype(np.float32), fs, onsets, cfg


def correct_native(x, reference, fs, volumes, cfg):
    data_up, fs_up, volumes_up = upsample(x[None, :], fs, volumes, cfg.interp_factor)
    ref_up, _, _ = upsample(reference[None, :], fs, volumes, cfg.interp_factor)
    markers, errors, length = compute_slice_markers(
        volumes_up, cfg.fixed_sdur_seconds, 0.0, fs_up, cfg.n_sg,
    )
    clean_up = _correct_channel(
        data_up[0], ref_up[0], markers, errors, length,
        build_slice_info(len(markers), cfg.n_sg, cfg.window_size),
        0, len(data_up[0]), 0, cfg, fs_up,
    )
    correction = downsample((data_up[0] - clean_up)[None, :], fs_up,
                            cfg.interp_factor, len(x))[0]
    return x - correction


def test_null_artifact_reference_preserves_native_samples_exactly():
    emg, _, fs, volumes, cfg = synthetic_emg(n_vol=8)
    clean = correct_native(emg, np.zeros_like(emg), fs, volumes, cfg)
    np.testing.assert_array_equal(clean, emg)


def test_unrelated_physiological_activity_is_not_template_subtracted():
    emg, _, fs, volumes, cfg = synthetic_emg()
    clean = correct_native(emg, hpf_fir(emg[None, :], fs, 30)[0], fs, volumes, cfg)
    np.testing.assert_allclose(clean, emg, atol=1e-7, rtol=0)


@pytest.mark.parametrize("pca_components", [0, 4])
def test_gradient_and_gap_templates_recover_independent_bursts(pca_components):
    emg, artifact, fs, volumes, cfg = synthetic_emg()
    cfg.pca_max_components = pca_components
    x = emg + artifact
    clean = correct_native(x, hpf_fir(x[None, :], fs, 30)[0], fs, volumes, cfg)
    error = np.linalg.norm(clean - emg) / np.linalg.norm(emg)
    gain = np.dot(clean, emg) / np.dot(emg, emg)
    assert error < 0.5, error
    assert 0.90 < gain < 1.10, gain
    # The volume gap contains both an artifact and physiological samples.
    phase = np.arange(len(emg)) / fs % cfg.tr
    gaps = phase >= cfg.n_sg * cfg.fixed_sdur_seconds
    assert np.linalg.norm((clean-emg)[gaps]) < 0.5*np.linalg.norm(emg[gaps])
    assert np.count_nonzero(clean[gaps]) == np.count_nonzero(gaps)


def test_fractional_template_roundtrip_has_no_internal_sample_holes():
    x = np.ones(200, dtype=np.float32)
    onsets = np.array([0, 31, 63, 94, 126])
    errors = np.array([0.0, 0.4, -0.2, 0.2, -0.4])
    stops = np.r_[onsets[1:], 157]
    segs, valid = extract_aligned_segments(x, onsets, 31, errors,
                                          padding=16, include_padding=True)
    restored = restore_artifact_segments(segs, onsets, errors, valid, stops, len(x), 16)
    np.testing.assert_allclose(restored[:157], 1, atol=1e-6)
    np.testing.assert_array_equal(restored[157:], 0)


def test_templates_reject_unrelated_candidates():
    rng = np.random.default_rng(9)
    segments = rng.standard_normal((32, 400)).astype(np.float32)
    info = build_slice_info(32, 1, 50)
    templates = build_artifact_templates(segments, np.arange(32), info, 12, 0, 400)
    np.testing.assert_array_equal(templates, 0)


def pca_mixture(similarity, max_components=4):
    fs, length, count = 2000.0, 400, 40
    t = np.arange(length*count) / fs
    reference = np.sin(2*np.pi*180*t).astype(np.float32)
    independent = np.sin(2*np.pi*120*t)
    polarity = np.repeat(np.tile([-1., 1.], count//2), length)
    physiological = ((similarity*reference + np.sqrt(1-similarity**2)*independent)*polarity).astype(np.float32)
    out = pca_cleanup(reference+physiological, reference, np.arange(count)*length,
                      length, np.arange(count), fs, 0, len(t), 0,
                      mean_corr_threshold=1.0, max_components=max_components)
    return out, physiological, (independent*polarity).astype(np.float32)


def test_pca_zero_disables_components_and_mean_removal():
    out, physiological, _ = pca_mixture(0.8, max_components=0)
    np.testing.assert_allclose(out, physiological, atol=2e-7, rtol=0)


def test_pca_preserves_weakly_artifact_correlated_structure():
    out, physiological, _ = pca_mixture(0.15)
    np.testing.assert_allclose(out, physiological, atol=2e-7, rtol=0)


def test_pca_removes_only_artifact_supported_part_of_mixed_component():
    out, _, independent = pca_mixture(0.8)
    gain = np.dot(out, independent) / np.dot(independent, independent)
    assert gain > 0.57  # independent component amplitude is 0.6 before cleanup


def test_coarse_timing_can_find_gap_longer_than_half_group():
    fs, tr, n_sg, n_vol = 2000.0, 1.6, 18, 6
    sdur = 0.082
    onsets = np.rint(np.arange(n_vol)*tr*fs).astype(np.int64)
    t = np.arange(int(n_vol*tr*fs)) / fs
    x = np.zeros_like(t)
    for volume in onsets/fs:
        for group in range(n_sg):
            tau = t - volume - group*sdur
            x += np.sin(2*np.pi*320*tau)*np.exp(-((tau-.03)/.008)**2)
    estimate, gap, _ = estimate_initial_timing(x, fs, onsets, n_sg, n_vol,
                                               max_gap_seconds=.2)
    assert abs(estimate-sdur) <= 1/(fs*n_sg)
    assert gap > tr/n_sg/2


def raw_with_events(times, descriptions=None, first_samp=0):
    raw = mne.io.RawArray(np.zeros((1, 12000)), mne.create_info(["EXT"], 1000.0),
                          first_samp=first_samp, verbose=False)
    raw.set_annotations(mne.Annotations(times, np.zeros(len(times)),
                                       descriptions or ["Response/R128"]*len(times)))
    return raw


def test_trigger_selection_is_exact_and_relative_to_recording():
    raw = raw_with_events([.2, 1.8, 3.4, .5],
                          ["Response/R128"]*3 + ["Response/R1280"], first_samp=700)
    onsets, count, _ = detect_volume_onsets(raw, "R128", 1.6)
    np.testing.assert_array_equal(onsets, [200, 1800, 3400])
    assert count == 3


def test_missing_trigger_is_rejected_before_processing():
    with pytest.raises(ValueError, match="Irregular volume triggers"):
        detect_volume_onsets(raw_with_events([.2, 1.8, 5.0]), "R128", 1.6)


def test_explicit_mask_records_the_final_gap():
    intervals = gap_mask_intervals(np.array([0, 20, 50, 70]), np.array([1, 3]),
                                   10, 1.0, scan_stop=100)
    assert intervals == [[30, 50], [80, 100]]


def test_metadata_is_strict_json(tmp_path):
    filename = export_metadata(str(tmp_path), "qc", {"values": np.array([np.nan, np.inf, 1.])})
    assert json.loads(open(filename).read()) == {"values": [None, None, 1.]}


def test_harmonic_metric_detects_line_removal_above_background():
    fs = 1000.
    t = np.arange(10000)/fs
    background = np.random.default_rng(7).standard_normal(len(t))*.1
    raw = background + 2*np.sin(2*np.pi*60*t)
    row = compute_gradient_harmonic_reduction(raw[None, :], background[None, :],
                                              fs, ["EXT"], .05)[0]
    assert row["excess_power_ratio"] > 100


def test_brainvision_export_preserves_channels_annotations_and_timing(tmp_path):
    rng = np.random.default_rng(21)
    original = rng.standard_normal((2, 2000))*1e-5
    raw = mne.io.RawArray(original, mne.create_info(["EXT", "AUX"], 1000., ["emg", "ecg"]), verbose=False)
    raw.set_annotations(mne.Annotations([.5, 1.3], [0., 0.], ["R128", "R128"]))
    replacement = original[:1]*.8
    path = export_brainvision(raw, replacement, [0], str(tmp_path), "roundtrip")
    reread = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    np.testing.assert_allclose(reread.get_data()[0], replacement[0], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(reread.get_data()[1], original[1], rtol=1e-6, atol=1e-12)
    events, _ = mne.events_from_annotations(reread, verbose=False)
    np.testing.assert_array_equal(events[:, 0], [500, 1300])


@pytest.mark.parametrize("first_samp", [0, 700])
def test_brainvision_marker_rounding_preserves_single_samples(tmp_path, first_samp):
    fs = 5000.
    positions = np.array([3, 6, 12, 24, 29, 48, 58, 93])
    durations = np.ones(len(positions), dtype=int)
    raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(["EXT"], fs, "emg"),
                          first_samp=first_samp, verbose=False)
    raw.set_annotations(mne.Annotations(positions/fs, durations/fs,
                                       ["Response/R128"]*len(positions)))
    path = export_brainvision(raw, raw.get_data(), [0], str(tmp_path), "sample_exact")
    reread = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    events, _ = mne.events_from_annotations(reread, verbose=False)
    np.testing.assert_array_equal(events[:, 0], positions)
    np.testing.assert_array_equal(np.rint(reread.annotations.duration*fs), durations)


@pytest.mark.parametrize("mask_fraction", [0.0, 1.0])
def test_full_pipeline_identity_and_explicit_mask(tmp_path, monkeypatch, mask_fraction):
    emg, _, fs, volumes, cfg = synthetic_emg(n_vol=6)
    prefix, suffix = 73, 150
    full = np.r_[np.ones(prefix)*2e-6, emg*1e-5, np.ones(suffix)*3e-6]
    raw = mne.io.RawArray(full[None, :], mne.create_info(["EXT"], fs, "emg"), verbose=False)
    raw.set_annotations(mne.Annotations((volumes+prefix)/fs, np.zeros(len(volumes)), ["R128"]*len(volumes)))
    input_path = export_brainvision(raw, raw.get_data(), [0], str(tmp_path), "input")
    monkeypatch.setattr("farm.workflow._estimate_interval_artifact",
                        lambda reference, *args: np.zeros_like(reference))
    cfg.vhdr_path = input_path
    cfg.output_dir = str(tmp_path / "output")
    cfg.zero_fill_gap_fraction = mask_fraction
    result = run_pipeline(cfg)
    metadata = json.loads(open(result["metadata_path"]).read())
    expected = result["data_full_raw"].copy()
    for start, stop in metadata["masked_intervals_samples_full"]:
        expected[:, start:stop] = 0
    np.testing.assert_array_equal(result["data_clean_full"], expected)
    assert bool(metadata["masked_intervals_samples_full"]) == bool(mask_fraction)
    assert metadata["scan_start_sample"] == prefix
