"""Pipeline configuration dataclass."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class FARMConfig:
    """All user-facing parameters for a single FARM run.

    Attributes
    ----------
    vhdr_path : str
        Path to the BrainVision ``.vhdr`` header file.
    tr : float
        Repetition time in seconds.
    n_slices : int
        Total number of EPI slices per volume.
    mb_factor : int
        Multiband / simultaneous multi-slice factor (1 = no MB).
    trigger : str
        Volume-onset marker label (e.g. ``"R128"``).
    ch_regex : str
        Regex matched against channel names to select EMG channels.
    interp_factor : int
        Upsampling factor applied before FARM correction.
    window_size : int
        Half-width of the sliding window used to gather template candidates.
    n_candidates : int
        Number of best-correlated candidates kept per template.
    n_volumes : int or None
        If set, only process the first *n_volumes* volumes.
    time_section : float
        Duration (s) of each PCA section.
    var_threshold : float
        Minimum explained-variance percentage to retain a PCA component.
    bandpass : tuple of float
        Band-pass limits (Hz) applied for diagnostic comparisons.
    hpf_cutoff : float
        High-pass filter cutoff (Hz) applied in preprocessing.
    lpf_cutoff : float
        Low-pass filter cutoff (Hz) applied in post-processing.
    padding : int
        Extra samples on each side for FFT phase-shift extraction.
    output_dir : str
        Directory where all outputs are written.
    plot : bool
        If *True*, produce diagnostic plots during the pipeline.
    """

    # ── Input ────────────────────────────────────────────────
    vhdr_path: str = ""

    # ── EPI sequence ─────────────────────────────────────────
    tr: float = 1.6
    n_slices: int = 54
    mb_factor: int = 3
    trigger: str = "R128"
    ch_regex: str = r"EXT|FLE"

    # ── Processing ───────────────────────────────────────────
    interp_factor: int = 10
    window_size: int = 50
    n_candidates: int = 12
    n_volumes: Optional[int] = None
    time_section: float = 60.0
    var_threshold: float = 5.0
    bandpass: Tuple[float, float] = (30, 250)
    hpf_cutoff: float = 30.0
    lpf_cutoff: float = 250.0
    padding: int = 10

    # ── Output ───────────────────────────────────────────────
    output_dir: str = "FARM_output"
    plot: bool = True

    # ── Derived ──────────────────────────────────────────────
    @property
    def n_sg(self) -> int:
        """Number of slice-artifact groups per volume."""
        return self.n_slices // self.mb_factor

    def validate(self) -> None:
        """Raise if the configuration is inconsistent."""
        assert self.vhdr_path, "vhdr_path must be set"
        assert self.tr > 0, f"TR must be positive, got {self.tr}"
        assert self.n_slices > 0, "n_slices must be positive"
        assert self.mb_factor >= 1, "mb_factor must be >= 1"
        assert self.n_slices % self.mb_factor == 0, (
            f"{self.n_slices} slices not divisible by MB={self.mb_factor}"
        )
        assert self.interp_factor >= 1, "interp_factor must be >= 1"