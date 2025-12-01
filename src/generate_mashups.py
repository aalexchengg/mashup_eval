# Author @abcheng. Main function for generating mashups.
import argparse
from src.generators.base_mashup_generator import BaseMashupGenerator
from src.matching.match import Match
import json
import os
from tqdm import tqdm
import yaml
import logging

logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-generator', type = str,
                        choices = ['identity', 'auto', 'naive'],
                        default = 'identity',
                        help = "Matcher strategy.")
    parser.add_argument('-matches', type = str,
                        required = True,
                        help = "Path to jsonl of generated matches")
    parser.add_argument('-out_dir', type = str,
                        default = "ignore",
                        help = "Output directory of matcher, if specified.")
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
    out_dir = args.out_dir if args.out_dir != "ignore" else None
    logger.info(f"Creating a {args.generator} generator...")
    generator = BaseMashupGenerator.create(args.generator, out_dir)
    matches_path = os.path.abspath(args.matches)
    logger.info(f"Reading matches from {matches_path}...")
    with open(matches_path, 'r') as json_file:
        matches = list(json_file)
    logger.info(f"Begin mashup generation...")
    for json_str in tqdm(matches, desc = "Generating Mashups..."):
        fields = json.loads(json_str)
        match = Match(**fields) # recreate the match object
        paths = []
        for song in match.songs:
            paths.append(f"{match.directory}/{song}")
        logger.info(f"Generating a song for match {match.id} from {matches_path}...")
        generator.generate(paths, match.id, match.layers)
        logger.info("Finished generation.")
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