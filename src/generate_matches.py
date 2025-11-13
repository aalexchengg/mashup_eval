# Author @abcheng. Main function for generating matches.
from matching.base_matcher import BaseMatcher
import argparse
"""
Generates matches based on the strategy and max size given by user.
Usage: 
python3 generate_matches.py -masher [DEFAULT: 'naive'] -max_size [DEFAULT: -1] -inp_dir [path to input directory] \
-out_dir [OPTIONAL path to output directory] -out_path [path/filename of resulting jsonl]
"""


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-masher', type = str,
                        choices = ['naive'],
                        default = 'naive',
                        help = "Matcher strategy.")
    parser.add_argument('-max_size', type = int,
                        default = -1,
                        help = "Maximum size of the output.")
    parser.add_argument('-inp_dir', type = str,
                        help = "Directory of all the audio samples.")
    parser.add_argument('-out_dir', type = str,
                        default = "ignore",
                        help = "Output directory of matcher, if specified.")
    parser.add_argument('-out_path', type = str,
                        help = "Output path of jsonl. Is populated in out_dir if specified.")
    return parser

def main(args):
    # create the matcher
    out_dir = args.out_dir if args.out_dir != "ignore" else None
    matcher = BaseMatcher.create(args.masher, out_dir)
    matcher.generate_matches(args.inp_dir, args.max_size, args.out_path)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)