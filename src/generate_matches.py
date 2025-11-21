# Author @abcheng. Main function for generating matches.
from src.matching.base_matcher import BaseMatcher
import yaml
import argparse
import os
import logging
"""
Generates matches based on the strategy and max size given by user.
"""
logger = logging.getLogger(__name__)

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
                        help = "Output directory of matcher, if specified. Will be created in out/")
    parser.add_argument('-stem_dir', type = str,
                        default = None,
                        help = "Directory of stem tracks, if specified. Relative path is ok.")
    parser.add_argument('-out_path', type = str,
                        default = "match_out",
                        help = "Output path of jsonl. Is populated in out_dir if specified.")
    parser.add_argument('-verbose', type = bool,
                        default = False,
                        action = argparse.BooleanOptionalAction,
                        help = "Whether to output INFO level logs.")
    parser.add_argument('-config', type = str,
                        default = None,
                        help = "accepts yaml config files as well.")
    return parser

def main(args):
    # turn on verbose mode if true.
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    # create the matcher
    out_dir = args.out_dir if args.out_dir != "default" else None
    # set directories to absolute paths.
    inp_dir = os.path.abspath(args.inp_dir)
    stem_dir = os.path.abspath(args.stem_dir) if args.stem_dir else None
    logger.info(f"Input directory path is {inp_dir}.")
    if stem_dir:
        logger.info(f"Stem directory exists, and path is at {stem_dir}.")
    logger.info(f"Creating a {args.matcher} matcher...")
    matcher = BaseMatcher.create(args.matcher, out_dir, stem_dir)
    # generate!
    logger.info(f"Generating matches with max size: {args.max_size} sort strategy {args.sort}.")
    matcher.generate_matches(inp_dir, args.max_size, args.out_path, args.sort)
    logger.info("All done.")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    # optional config override.
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Overwrite args with config values
        for key, value in config.items():
            setattr(args, key, value)
    main(args)