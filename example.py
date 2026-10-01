"""Minimal single-run example for the validated 54-slice / MB3 protocol."""
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
    # This acquisition shows a strong scanner-locked inter-volume boundary
    # artifact; the complete measured gap is therefore masked.
    zero_fill_gap_fraction=1.0,
    output_dir="denoised",
)

result = run_pipeline(cfg)
print(result["brainvision_path"])
