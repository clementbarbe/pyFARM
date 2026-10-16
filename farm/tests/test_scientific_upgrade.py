"""Regression tests for opt-in pyFARM scientific upgrades."""
import numpy as np

from farm.config import FARMConfig
from farm.templates.robust import aggregate_template
from farm.diagnostics.gapmask import gap_mask_native


def test_historical_mean_and_trimmed_mean():
    x = np.array([[1., 2.], [3., 4.], [5., 100.]])
    np.testing.assert_array_equal(aggregate_template(x), x.mean(axis=0))
    np.testing.assert_array_equal(aggregate_template(x, trim_fraction=0.34),
                                  np.array([3., 4.]))


def test_robust_aggregations_resist_outlier():
    x = np.array([[1.], [1.], [1.], [1.], [101.]])
    assert aggregate_template(x, method="median")[0] == 1.0
    assert aggregate_template(x, method="huber")[0] < aggregate_template(x)[0]


def test_gap_mask_off_by_default():
    cfg = FARMConfig(vhdr_path="dummy.vhdr")
    cfg.validate()
    assert cfg.template_method == "mean"
    assert cfg.pca_mode == "current"
    assert cfg.timing_dtime_max_ms is None
    assert not cfg.export_gap_mask
    mask = gap_mask_native(np.array([0, 20, 50, 70]), np.array([1]),
                           10, 100, 1, 0.0)
    assert not np.any(mask)


def test_gap_mask_marks_true_intervolume_gap():
    mask = gap_mask_native(np.array([0, 20, 50, 70]), np.array([1]),
                           10, 100, 1, 1.0, guard_samples=0)
    assert np.all(mask[30:50])
    assert not np.any(mask[:30])
    assert not np.any(mask[50:])
