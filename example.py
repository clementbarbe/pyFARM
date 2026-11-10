"""Minimal Python example for a 1.660-s, 60-slice, MB3 acquisition."""
import argparse
from farm.config import FARMConfig
from farm.workflow import run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vhdr')
    parser.add_argument('--output-dir', default='output/example')
    args = parser.parse_args()
    run_pipeline(FARMConfig(vhdr_path=args.vhdr, output_dir=args.output_dir,
                            tr=1.660, n_slices=60, mb_factor=3, trigger='R128',
                            ch_regex='ZYG|COR', drop_last_volume=True,
                            science_mode='matched', timing_mode='initial',
                            zero_fill_gap_fraction=0., line_cleanup='off'))
