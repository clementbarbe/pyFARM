import numpy as np

from farm.config import FARMConfig
from farm.pca.cleanup import pca_cleanup
from farm.templates.subtraction import build_artifact_templates
from farm.templates.zerofill import zero_fill_dtime


def test_default_does_not_zero_intervolume_gap():
    cfg = FARMConfig(vhdr_path="dummy.vhdr")
    assert cfg.zero_fill_gap_fraction == 0.0

    x = np.ones(100, dtype=np.float32)
    onsets = np.array([0, 20, 50, 70], dtype=np.int64)
    out = zero_fill_dtime(
        x, onsets, last_slice_idx=np.array([1]), seg_len=10,
        gap_fraction=cfg.zero_fill_gap_fraction,
    )
    np.testing.assert_array_equal(out, x)


def test_last_group_template_uses_full_segment_not_deadtime_shortened_window():
    seg_len = 64
    # Target and two candidates agree strongly over the full segment but not
    # in the first 8 samples.  A legacy dtime-shortened window would reject it.
    base = np.sin(np.linspace(0, 6 * np.pi, seg_len)).astype(np.float32)
    target = base.copy()
    target[:8] *= -1
    c1 = base.copy()
    c2 = (0.95 * base).astype(np.float32)
    aligned = np.stack([target, c1, c2])
    valid = np.array([0, 1, 2], dtype=np.int64)
    slice_info = {
        "candidate_idx": np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
        "is_last": np.array([True, False, False]),
    }
    artifact = build_artifact_templates(
        aligned, valid, slice_info, n_candidates=2,
        dtime_samp_up=1000, seg_len=seg_len,
        min_correlation=0.2, trim_fraction=0.0,
    )
    assert np.linalg.norm(artifact[0]) > 0.1


def test_pca_does_not_remove_structure_without_artifact_reference():
    rng = np.random.default_rng(42)
    srate = 1000.0
    seg_len = 100
    n_seg = 12
    onsets = np.arange(n_seg, dtype=np.int64) * seg_len
    # Repeated high-frequency structure could look PCA-dominant, but with no
    # scanner-artifact reference it must be preserved by the conservative gate.
    t = np.arange(seg_len) / srate
    burst = np.sin(2 * np.pi * 120 * t).astype(np.float32)
    x = np.tile(burst, n_seg) + 0.01 * rng.standard_normal(n_seg * seg_len)
    x = x.astype(np.float32)
    noise = np.zeros_like(x)
    clean = pca_cleanup(
        x, noise, onsets, seg_len, np.arange(n_seg), srate,
        0, len(x), 0, time_section=60, var_threshold=1.0,
        n_sg=1, artifact_corr_threshold=0.05,
        mean_corr_threshold=0.05, mean_repeatability_threshold=0.0,
        max_components=6, groupwise=True,
    )
    np.testing.assert_allclose(clean, x, rtol=0, atol=1e-6)
