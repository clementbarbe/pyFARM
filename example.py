"""Minimal single-run example for a 54-slice / MB3 protocol."""
from farm.config import FARMConfig
from farm.workflow import run_pipeline

cfg = FARMConfig(
    vhdr_path="data/me3mb3_tr1600_sl54.vhdr",
    tr=1.6,
    n_slices=54,
    mb_factor=3,
    trigger="R128",
    ch_regex=r"EXT|FLE",
    drop_last_volume=True,
    # Dedicated gap templates preserve measured EMG between volumes.
    correct_volume_gaps=True,
    zero_fill_gap_fraction=0.0,
    output_dir="denoised",
)

result = run_pipeline(cfg)
print(result["brainvision_path"])
