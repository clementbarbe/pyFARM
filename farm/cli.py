"""Command-line interface for pure pyFARM denoising."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sys
import traceback

from farm.config import FARMConfig
from farm.workflow import run_pipeline


signal.signal(signal.SIGPIPE, signal.SIG_DFL)


class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except BrokenPipeError:
            pass


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tr", type=float, required=True, help="Repetition time in seconds")
    p.add_argument("--n-slices", type=int, required=True, help="Number of anatomical EPI slices")
    p.add_argument("--mb-factor", type=int, default=1, help="Multiband factor")
    p.add_argument("--trigger", default="R128", help="Volume trigger label")
    p.add_argument("--ch-regex", default=r"EXT|FLE", help="Regex selecting EMG channels")
    p.add_argument("--interp-factor", type=int, default=10)
    p.add_argument("--window-size", type=int, default=50, help="Template search window in volumes")
    p.add_argument("--n-candidates", type=int, default=12)
    p.add_argument("--n-volumes", type=int, default=None)
    p.add_argument("--drop-last-volume", action="store_true")
    p.add_argument("--time-section", type=float, default=60.0)
    p.add_argument("--var-threshold", type=float, default=5.0)
    p.add_argument("--artifact-hpf", type=float, default=30.0)
    p.add_argument("--timing-hpf", type=float, default=30.0)
    p.add_argument("--template-min-corr", type=float, default=-1.0)
    p.add_argument("--template-trim", type=float, default=0.0)
    p.add_argument("--pca-artifact-corr", type=float, default=0.10)
    p.add_argument("--pca-max-components", type=int, default=6)
    p.add_argument(
        "--zero-fill-gap-fraction", type=float, default=0.0,
        help="Fraction of true inter-volume gap masked (validated dataset: 1.0)",
    )
    p.add_argument("--qc-band", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(30.0, 250.0))
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--figures-dir", default=None)
    p.add_argument("--suffix", default="_FARM", help="Output basename suffix")


def _make_config(vhdr: Path, output_dir: Path, args) -> FARMConfig:
    return FARMConfig(
        vhdr_path=str(vhdr),
        tr=args.tr,
        n_slices=args.n_slices,
        mb_factor=args.mb_factor,
        trigger=args.trigger,
        ch_regex=args.ch_regex,
        interp_factor=args.interp_factor,
        window_size=args.window_size,
        n_candidates=args.n_candidates,
        n_volumes=args.n_volumes,
        drop_last_volume=args.drop_last_volume,
        time_section=args.time_section,
        var_threshold=args.var_threshold,
        artifact_hpf_cutoff=args.artifact_hpf,
        timing_hpf_cutoff=args.timing_hpf,
        template_min_correlation=args.template_min_corr,
        template_trim_fraction=args.template_trim,
        pca_artifact_corr_threshold=args.pca_artifact_corr,
        pca_max_components=args.pca_max_components,
        zero_fill_gap_fraction=args.zero_fill_gap_fraction,
        qc_bandpass=tuple(args.qc_band),
        output_dir=str(output_dir),
        figures_dir="" if args.no_figures else args.figures_dir,
        output_suffix=args.suffix,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyfarm-denoise",
        description="Pure FARM-inspired EMG-fMRI gradient-artifact denoising.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Denoise one BrainVision recording")
    run.add_argument("vhdr", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("denoised"))
    _add_common(run)

    batch = sub.add_parser("batch", help="Recursively denoise a dataset tree")
    batch.add_argument("input_root", type=Path)
    batch.add_argument("output_root", type=Path)
    batch.add_argument("--pattern", default="*.vhdr", help="Recursive input glob")
    batch.add_argument("--fail-fast", action="store_true")
    batch.add_argument("--skip-existing", action="store_true")
    _add_common(batch)
    return p


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = _SafeStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def _run_one(vhdr: Path, output_dir: Path, args) -> dict:
    if not vhdr.exists():
        raise FileNotFoundError(vhdr)
    return run_pipeline(_make_config(vhdr, output_dir, args))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "run":
        _run_one(args.vhdr, args.output_dir, args)
        return 0

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    files = sorted(p.resolve() for p in input_root.rglob(args.pattern))
    # Freeze discovery before creating outputs; also avoid reprocessing FARM files.
    files = [p for p in files if args.suffix not in p.stem]
    if not files:
        raise SystemExit(f"No BrainVision files matching {args.pattern!r} under {input_root}")

    logging.getLogger("farm.cli").info("Discovered %d runs", len(files))
    failures: list[tuple[Path, str]] = []
    for i, vhdr in enumerate(files, 1):
        rel_parent = vhdr.parent.relative_to(input_root)
        out_dir = output_root / rel_parent
        out_vhdr = out_dir / f"{vhdr.stem}{args.suffix}.vhdr"
        if args.skip_existing and out_vhdr.exists():
            logging.getLogger("farm.cli").info(
                "[%d/%d] skip existing %s", i, len(files), out_vhdr
            )
            continue
        logging.getLogger("farm.cli").info(
            "[%d/%d] %s -> %s", i, len(files), vhdr, out_dir
        )
        try:
            _run_one(vhdr, out_dir, args)
        except Exception as exc:  # batch should report all failures by default
            failures.append((vhdr, f"{type(exc).__name__}: {exc}"))
            logging.getLogger("farm.cli").error("Failed %s: %s", vhdr, exc)
            if args.verbose:
                traceback.print_exc()
            if args.fail_fast:
                raise

    if failures:
        logging.getLogger("farm.cli").error("%d/%d runs failed", len(failures), len(files))
        for path, msg in failures:
            logging.getLogger("farm.cli").error("  %s — %s", path, msg)
        return 1
    logging.getLogger("farm.cli").info("All %d runs completed", len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
