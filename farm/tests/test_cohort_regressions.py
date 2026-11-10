"""Failures found in cohort data: inconsistent filters, ADC rails, timing refs."""

import numpy as np
from farm.diagnostics.acquisition import assess_adc_rails
from farm.preprocessing.filters import hpf_fir, apply_bandpass
from farm.timing.estimation import timing_consensus
from farm.config import FARMConfig
from farm.tests.test_preservation_regressions import correct_native


def test_matching_science_filter_removes_periodic_transition_band_residual():
    fs, tr, sdur, count = 1000., .4, .08, 80
    t = np.arange(int(count*tr*fs))/fs
    raw = (1e-3*np.sin(2*np.pi*25*t) + 3e-4*np.sin(2*np.pi*37.5*t)).astype(np.float32)
    reference = hpf_fir(raw[None, :], fs, 30)[0]
    volumes = np.rint(np.arange(count)*tr*fs).astype(np.int64)
    cfg = FARMConfig(vhdr_path="dummy.vhdr", tr=tr, n_slices=4, mb_factor=1,
                     fixed_sdur_seconds=sdur, timing_mode="fixed", interp_factor=4,
                     pca_max_components=0)
    legacy = correct_native(raw, reference, fs, volumes, cfg)
    matched = correct_native(reference, reference, fs, volumes, cfg)
    legacy = apply_bandpass(legacy, fs, (30,250))[1000:-1000]
    matched = apply_bandpass(matched, fs, (30,250))[1000:-1000]
    assert np.linalg.norm(matched) < .05*np.linalg.norm(legacy)


def test_symmetric_adc_rails_and_volume_indices(tmp_path):
    source = tmp_path/"raw.vhdr"
    source.write_text("[Binary Infos]\nBinaryFormat=INT_16\n")
    scale = .5e-6
    codes = np.array([-32768,-32767,0,100,32767,32767,32767,0], dtype=np.int32)
    report = assess_adc_rails((codes*scale).astype(np.float32)[None, :], 1000.,
                              [{"cal":.5, "range":1e-6}], ["COR"], [0,4], source)[0]
    assert report["rail_samples"] == 5
    assert report["affected_volume_indices"] == [0,1]
    assert report["rail_samples_by_volume"] == [2,3]
    assert report["longest_rail_seconds"] == .003


def test_float_export_is_not_declared_free_of_adc_clipping(tmp_path):
    source = tmp_path/"export.vhdr"
    source.write_text("[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n")
    report = assess_adc_rails(np.zeros((1,10)),1000.,[{"cal":1,"range":1}],
                              ["COR"],[0],source)[0]
    assert not report["assessed"]
    assert "rail_samples" not in report


def test_dead_timing_channel_does_not_pull_reliable_reference_to_midpoint():
    duration, report = timing_consensus([.08157,.082],[.409,19.906],5000.,20)
    assert duration == .082
    assert report["channels_used"] == [1]
    assert not report["unresolved_disagreement"]


def test_quantization_sized_timing_variation_still_uses_both_channels():
    duration, report = timing_consensus([.08199,.082],[5.,20.],5000.,20)
    assert np.isclose(duration,.081995)
    assert report["channels_used"] == [0,1]


def test_conflicting_reliable_timing_channels_are_marked_unresolved():
    _, report = timing_consensus([.08157,.082],[18.,20.],5000.,20)
    assert report["unresolved_disagreement"]
