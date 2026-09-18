"""Pipeline configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math


@dataclass
class FARMConfig:
    """All user-facing parameters for a single FARM run."""

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

    # window_size is now in VOLUMES, not in arbitrary global slices.
    # For each slice-group, templates can use ±window_size nearby volumes.
    window_size: int = 50

    n_candidates: int = 12
    n_volumes: Optional[int] = None
    drop_last_volume: bool = False

    time_section: float = 60.0
    var_threshold: float = 5.0

    bandpass: Tuple[float, float] = (30.0, 250.0)
    hpf_cutoff: float = 30.0
    lpf_cutoff: float = 250.0
    padding: int = 10

    # Fraction of the actual inter-volume gap to suppress around
    # the volume boundary. 0.0 means no masking; 1.0 masks all gap.
    zero_fill_gap_fraction: float = 1.0

    # ── Envelope baseline correction ─────────────────────────
    envelope_baseline: str = "robust"
    envelope_percentile: float = 10.0
    envelope_window_sec: float = 30.0
    envelope_threshold_factor: float = 2.5

    # ── HRF centering + log compression ──────────────────────
    center_hrf: bool = True
    log_compress_gain: float = 50.0

    # ── Output ───────────────────────────────────────────────
    output_dir: str = "output"
    figures_dir: Optional[str] = None

    @property
    def n_sg(self) -> int:
        """Number of acquisition slice-groups per volume.

        ceil() deliberately supports protocols such as:
        - 53 anatomical slices
        - multiband factor 3
        - 18 slice-group acquisition events

        The final multiband group may contain fewer anatomical slices, but
        it remains one acquisition event and therefore one gradient artifact.
        """
        return int(math.ceil(self.n_slices / self.mb_factor))

    @property
    def figures_enabled(self) -> bool:
        return self.figures_dir != ""

    def get_figures_dir(self) -> Path | None:
        """Return/create figures directory, or None if figures are disabled."""
        if self.figures_dir == "":
            return None

        if self.figures_dir is None:
            p = Path(self.output_dir) / "figures"
        else:
            p = Path(self.figures_dir)

        p.mkdir(parents=True, exist_ok=True)
        return p

    def validate(self) -> None:
        """Validate configuration independent of the actual input sampling rate."""
        if not self.vhdr_path:
            raise ValueError("vhdr_path must be set")

        if self.tr <= 0:
            raise ValueError(f"tr must be positive, got {self.tr}")

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

        if self.padding < 0:
            raise ValueError("padding must be >= 0")

        if not (0.0 <= self.zero_fill_gap_fraction <= 1.0):
            raise ValueError("zero_fill_gap_fraction must be in [0, 1]")

        if self.envelope_baseline not in ("none", "percentile", "robust"):
            raise ValueError(
                "envelope_baseline must be 'none', 'percentile' or 'robust', "
                f"got '{self.envelope_baseline}'"
            )

        if self.log_compress_gain < 0:
            raise ValueError("log_compress_gain must be >= 0")

        if len(self.bandpass) != 2:
            raise ValueError("bandpass must be a (low_hz, high_hz) tuple")

        if not (0 < self.bandpass[0] < self.bandpass[1]):
            raise ValueError(
                f"Invalid bandpass={self.bandpass}; require 0 < low < high."
            )

    def validate_sampling_rate(self, srate: float) -> None:
        """Validate frequency-dependent configuration after loading the data."""
        if srate <= 0:
            raise ValueError(f"Invalid sampling rate: {srate}")

        nyquist = srate / 2.0

        checks = {
            "hpf_cutoff": self.hpf_cutoff,
            "lpf_cutoff": self.lpf_cutoff,
            "bandpass low": self.bandpass[0],
            "bandpass high": self.bandpass[1],
        }

        for name, freq in checks.items():
            if not (0.0 < freq < nyquist):
                raise ValueError(
                    f"{name}={freq} Hz must be strictly between 0 and "
                    f"Nyquist ({nyquist:.3f} Hz)."
                )

        if self.hpf_cutoff >= self.lpf_cutoff:
            raise ValueError(
                f"hpf_cutoff ({self.hpf_cutoff}) must be below "
                f"lpf_cutoff ({self.lpf_cutoff})."
            )

        if self.bandpass[0] >= self.bandpass[1]:
            raise ValueError("bandpass low cutoff must be below high cutoff.")