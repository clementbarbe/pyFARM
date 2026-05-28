#!/usr/bin/env python
"""
farm — Command-line entry point.

Usage:
    python run.py path/to/data.vhdr --tr 1.6 --n-slices 54 --mb-factor 3
    python run.py path/to/data.vhdr --tr 1.6 --n-slices 54 --no-figures
"""

import argparse
import logging
import sys


def parse_args():
    """Parse command-line arguments into a FARMConfig."""
    p = argparse.ArgumentParser(
        prog="farm",
        description="FARM EMG-fMRI gradient artifact removal pipeline.",
    )
    p.add_argument("vhdr", help="Path to BrainVision .vhdr file")
    p.add_argument("--tr", type=float, required=True, help="Repetition time (s)")
    p.add_argument("--n-slices", type=int, required=True, help="Total EPI slices")
    p.add_argument("--mb-factor", type=int, default=1,
                    help="Multiband factor (default 1)")
    p.add_argument("--trigger", default="R128",
                    help="Volume trigger label (default R128)")
    p.add_argument("--ch-regex", default=r"EXT|FLE",
                    help="Regex for EMG channel names")
    p.add_argument("--output-dir", default="output",
                    help="Output directory")
    p.add_argument("--figures-dir", default=None,
                    help="Figure output directory (default: <output-dir>/figures)")
    p.add_argument("--no-figures", action="store_true",
                    help="Disable diagnostic figure generation entirely")
    p.add_argument("--interp-factor", type=int, default=10,
                    help="Upsampling factor")
    p.add_argument("--window-size", type=int, default=50,
                    help="Template candidate window")
    p.add_argument("--n-candidates", type=int, default=12,
                    help="Template candidates kept")
    p.add_argument("--n-volumes", type=int, default=None,
                    help="Limit number of volumes")
    p.add_argument("--time-section", type=float, default=60.0,
                    help="PCA section length (s)")
    p.add_argument("--var-threshold", type=float, default=5.0,
                    help="PCA variance threshold (%%)")
    p.add_argument("--verbose", action="store_true",
                    help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    from farm.config import FARMConfig
    from farm.workflow import run_pipeline

    # Handle --no-figures → set figures_dir to "" to disable
    if args.no_figures:
        figures_dir = ""
    else:
        figures_dir = args.figures_dir  # None → auto, or user-specified path

    cfg = FARMConfig(
        vhdr_path=args.vhdr,
        tr=args.tr,
        n_slices=args.n_slices,
        mb_factor=args.mb_factor,
        trigger=args.trigger,
        ch_regex=args.ch_regex,
        output_dir=args.output_dir,
        figures_dir=figures_dir,
        interp_factor=args.interp_factor,
        window_size=args.window_size,
        n_candidates=args.n_candidates,
        n_volumes=args.n_volumes,
        time_section=args.time_section,
        var_threshold=args.var_threshold,
    )

    results = run_pipeline(cfg)
    logging.getLogger("farm").info("Pipeline complete.")
    return results


if __name__ == "__main__":
    main()