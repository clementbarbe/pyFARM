"""Configuration for the pure pyFARM denoising pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math


@dataclass
class FARMConfig:
    """User-facing parameters for one EMG-fMRI denoising run.

    The package deliberately stops at denoising.  Envelope extraction,
    normalisation, HRF convolution, and SPM regressor construction belong to
    the separate ``emg-regressors`` tool.
    """

    vhdr_path: str = ""

    # EPI sequence
    tr: float = 1.6
    n_slices: int = 54
    mb_factor: int = 3
    trigger: str = "R128"
    ch_regex: str = r"EXT|FLE"

    # FARM processing
    interp_factor: int = 10
    window_size: int = 50       # candidate window in volumes
    n_candidates: int = 12
    n_volumes: Optional[int] = None
    drop_last_volume: bool = False
    time_section: float = 60.0
    var_threshold: float = 5.0

    # Artifact / timing branches.  The validated protocol uses the same HPF
    # for both, reproducing the stable timing minimum of the legacy pipeline.
    artifact_hpf_cutoff: float = 30.0
    timing_hpf_cutoff: float = 30.0

    # Artifact-specific global PCA
    pca_artifact_corr_threshold: float = 0.10
    pca_mean_corr_threshold: float = 0.10
    pca_mean_repeatability_threshold: float = 0.20
    pca_max_components: int = 6
    pca_groupwise: bool = False

    # Adaptive template construction.  These defaults preserve the validated
    # FARM-like top-N arithmetic mean while keeping the corrected last-group
    # correlation window.
    template_trim_fraction: float = 0.0
    template_min_correlation: float = -1.0
    template_scale_bounds: Optional[Tuple[float, float]] = None

    padding: int = 10

    # Generic default is conservative.  For protocols with a demonstrated
    # scanner-locked boundary artifact (such as the supplied dataset), set 1.0.
    zero_fill_gap_fraction: float = 0.0

    # QC band only.  It NEVER modifies the exported denoised waveform.
    qc_bandpass: Tuple[float, float] = (30.0, 250.0)

    # Output
    output_dir: str = "denoised"
    figures_dir: Optional[str] = None
    output_suffix: str = "_FARM"

    @property
    def n_sg(self) -> int:
        """Number of multiband acquisition groups per volume."""
        return int(math.ceil(self.n_slices / self.mb_factor))

    @property
    def figures_enabled(self) -> bool:
        return self.figures_dir != ""

    def get_figures_dir(self, basename: str | None = None) -> Path | None:
        if self.figures_dir == "":
            return None
        if self.figures_dir is None:
            p = Path(self.output_dir) / "figures"
            if basename:
                p = p / basename
        else:
            p = Path(self.figures_dir)
            if basename:
                p = p / basename
        p.mkdir(parents=True, exist_ok=True)
        return p

    def validate(self) -> None:
        if not self.vhdr_path:
            raise ValueError("vhdr_path must be set")
        if self.tr <= 0:
            raise ValueError("tr must be positive")
        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive")
        if self.mb_factor < 1:
            raise ValueError("mb_factor must be >= 1")
        if self.interp_factor < 1:
            raise ValueError("interp_factor must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1 volume")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if self.time_section <= 0:
            raise ValueError("time_section must be positive")
        if not (0.0 < self.var_threshold <= 100.0):
            raise ValueError("var_threshold must be in ]0, 100]")
        if not (0.0 <= self.pca_artifact_corr_threshold <= 1.0):
            raise ValueError("pca_artifact_corr_threshold must be in [0, 1]")
        if not (0.0 <= self.pca_mean_corr_threshold <= 1.0):
            raise ValueError("pca_mean_corr_threshold must be in [0, 1]")
        if self.pca_mean_repeatability_threshold < 0.0:
            raise ValueError("pca_mean_repeatability_threshold must be >= 0")
        if self.pca_max_components < 0:
            raise ValueError("pca_max_components must be >= 0")
        if not (0.0 <= self.template_trim_fraction < 0.5):
            raise ValueError("template_trim_fraction must be in [0, 0.5)")
        if not (-1.0 <= self.template_min_correlation <= 1.0):
            raise ValueError("template_min_correlation must be in [-1, 1]")
        if self.template_scale_bounds is not None:
            if len(self.template_scale_bounds) != 2 or not (
                self.template_scale_bounds[0] < self.template_scale_bounds[1]
            ):
                raise ValueError("template_scale_bounds must be None or (min, max)")
        if self.padding < 0:
            raise ValueError("padding must be >= 0")
        if not (0.0 <= self.zero_fill_gap_fraction <= 1.0):
            raise ValueError("zero_fill_gap_fraction must be in [0, 1]")
        if len(self.qc_bandpass) != 2 or not (
            0.0 < self.qc_bandpass[0] < self.qc_bandpass[1]
        ):
            raise ValueError("qc_bandpass must be (low, high) with 0 < low < high")
        if not self.output_suffix:
            raise ValueError("output_suffix must not be empty")

    def validate_sampling_rate(self, srate: float) -> None:
        if srate <= 0:
            raise ValueError(f"Invalid sampling rate: {srate}")
        nyquist = srate / 2.0
        for name, freq in {
            "artifact_hpf_cutoff": self.artifact_hpf_cutoff,
            "timing_hpf_cutoff": self.timing_hpf_cutoff,
            "qc_band low": self.qc_bandpass[0],
            "qc_band high": self.qc_bandpass[1],
        }.items():
            if not (0.0 < freq < nyquist):
                raise ValueError(
                    f"{name}={freq} Hz must be strictly between 0 and Nyquist "
                    f"({nyquist:.3f} Hz)."
                )
