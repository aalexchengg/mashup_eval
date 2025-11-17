# Author @abcheng. Main function for generating matches.
from matching.base_matcher import BaseMatcher
import argparse
import os
"""
Generates matches based on the strategy and max size given by user.
"""


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-matcher', type = str,
                        choices = ['naive', 'cocola'],
                        default = 'naive',
                        help = "Matcher strategy.")
    parser.add_argument('-sort', type = str,
                        choices = ['largest', 'smallest', 'unsorted'],
                        default = 'unsorted',
                        help = "Sort strategy.")
    parser.add_argument('-max_size', type = int,
                        default = -1,
                        help = "Maximum size of the output.")
    parser.add_argument('-inp_dir', type = str,
                        required = True,
                        help = "Directory of all the audio samples. Relative path is ok.")
    parser.add_argument('-out_dir', type = str,
                        default = "default",
                        help = "Output directory of matcher, if specified.")
    parser.add_argument('-stem_dir', type = str,
                        default = None,
                        help = "Directory of stem tracks, if specified. Relative path is ok.")
    parser.add_argument('-out_path', type = str,
                        default = "match_out",
                        help = "Output path of jsonl. Is populated in out_dir if specified.")
    return parser

def main(args):
    # create the matcher
    out_dir = args.out_dir if args.out_dir != "default" else None
    # set directories to absolute paths.
    inp_dir = os.path.abspath(args.inp_dir)
    stem_dir = os.path.abspath(args.stem_dir) if stem_dir else None
    matcher = BaseMatcher.create(args.masher, out_dir, stem_dir)
    # generate!
    matcher.generate_matches(inp_dir, args.max_size, args.out_path)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)