import argparse
from generators.base_mashup_generator import BaseMashupGenerator

def setup_parser():
    parser = argparse.ArgumentParser()
    # TODO: add parser args
    return parser

def main(args):
    pass


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)