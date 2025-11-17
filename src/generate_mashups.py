# Author @abcheng. Main function for generating mashups.
import argparse
from generators.base_mashup_generator import BaseMashupGenerator
from matching.match import Match
import json

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-generator', type = str,
                        choices = ['identity', 'auto'],
                        default = 'identity',
                        help = "Matcher strategy.")
    parser.add_argument('-matches', type = str,
                        required = True,
                        help = "Path to jsonl of generated matches")
    parser.add_argument('-out_dir', type = str,
                        default = "ignore",
                        help = "Output directory of matcher, if specified.")
    return parser

def main(args):
    out_dir = args.out_dir if args.out_dir != "ignore" else None
    generator = BaseMashupGenerator.create(args.generator, out_dir)
    with open(args.matches, 'r') as json_file:
        matches = list(json_file)
    for json_str in matches:
        fields = json.loads(json_str)
        match = Match(**fields) # recreate the match object
        paths = []
        for song in match.songs:
            paths.append(f"{match.directory}/{song}")
        generator.generate(paths, match.id)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)