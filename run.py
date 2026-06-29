#!/usr/bin/env python
"""
FARMICHE — Command-line entry point.
"""

import argparse
import logging
import signal
import sys


# ── Handle broken pipe gracefully (stdout closed early) ──────
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that silently ignores BrokenPipeError."""

    def emit(self, record):
        try:
            super().emit(record)
        except BrokenPipeError:
            pass

    def flush(self):
        try:
            super().flush()
        except BrokenPipeError:
            pass


def parse_args():
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
    p.add_argument("--ch-regex", default="EXT|FLE",
                    help="Regex for EMG channel names")
    p.add_argument("--output-dir", default="FARM_output",
                    help="Output directory")
    p.add_argument("--figures-dir", default=None,
                    help="Figure output directory (default: <output-dir>/figures)")
    p.add_argument("--no-figures", action="store_true",
                    help="Disable diagnostic figure generation")
    p.add_argument("--interp-factor", type=int, default=10,
                    help="Upsampling factor")
    p.add_argument("--window-size", type=int, default=50,
                    help="Template candidate window")
    p.add_argument("--n-candidates", type=int, default=12,
                    help="Template candidates kept")
    p.add_argument("--n-volumes", type=int, default=None,
                    help="Limit number of volumes")
    p.add_argument("--drop-last-volume", action="store_true",
                    help="Remove the last volume (manual acquisition stop)")
    p.add_argument("--time-section", type=float, default=60.0,
                    help="PCA section length (s)")
    p.add_argument("--var-threshold", type=float, default=5.0,
                    help="PCA variance threshold (%%)")
    p.add_argument("--envelope-baseline", default="robust",
                    choices=["none", "percentile", "robust"],
                    help="Envelope baseline correction method")
    p.add_argument("--envelope-percentile", type=float, default=10.0,
                    help="Percentile for rolling baseline")
    p.add_argument("--envelope-window", type=float, default=30.0,
                    help="Rolling baseline window (s)")
    p.add_argument("--envelope-threshold", type=float, default=2.5,
                    help="Noise threshold factor")
    p.add_argument("--no-center-hrf", action="store_true",
                    help="Disable HRF baseline centering")
    p.add_argument("--log-compress-gain", type=float, default=50.0,
                    help="Log compression gain (0 = legacy)")
    p.add_argument("--verbose", action="store_true",
                    help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    handler = _SafeStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    from farm.config import FARMConfig
    from farm.workflow import run_pipeline

    figures_dir = "" if args.no_figures else args.figures_dir

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
        drop_last_volume=args.drop_last_volume,
        time_section=args.time_section,
        var_threshold=args.var_threshold,
        envelope_baseline=args.envelope_baseline,
        envelope_percentile=args.envelope_percentile,
        envelope_window_sec=args.envelope_window,
        envelope_threshold_factor=args.envelope_threshold,
        center_hrf=not args.no_center_hrf,
        log_compress_gain=args.log_compress_gain,
    )

    try:
        results = run_pipeline(cfg)
    except BrokenPipeError:
        pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except BrokenPipeError:
        os_devnull = open("/dev/null", "w")
        sys.stdout = os_devnull
        sys.stderr = os_devnull
        sys.exit(0)