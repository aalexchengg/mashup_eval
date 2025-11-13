# Author: @echu2
# Main file for running metric suite

import argparse
import librosa
import tqdm
import numpy as np
import os
from src.evaluate.evaluate import EmbeddingGenerator
from scipy.spatial.distance import pdist

"""
NOTES:
    we want holdout set to be modular: currently this script will take in some directory of wav files
    so that we can just assume our holdout set is some unknown group of wav files.

    for each holdout set, we can/should make a class that holds the following info:
    - location of global centroid ([min_timesteps, 768])
    - value of diameter (float, euclidean distance of furthest pair)
    - number of holdout audio files
    - list ? dictionary ? of each eval example's embeddings

    it should have the following capabilities:
    - given an audio file, compute its embedding
    - given an embedding of a generation (a test audio file of min_timesteps),
      return distance to nearest neighbor in holdout set
    - given an embedding of a generation (a test audio file of min_timesteps),
      return distance to centroid of holdout set

    ASSUMPTIONS:
    - all wav files are the same length. embedding generator makes [timestep, 768] embeddings so we can simply
      truncate to the min length.


    we're going to make a wrapper / another class that takes in a "full" generation (i.e. a generation of a 
    non-fixed length) and use the holdout set distances in some specific way.

"""


class HoldoutSet:

    def __init__(self, list_path, embed_model_name=None):
        self.list_path = list_path
        self.audio_files = dict()
        if embed_model_name:
            self.embedding_generator = EmbeddingGenerator(embed_model_name)
        else:
            self.embedding_generator = EmbeddingGenerator()

        for f in tqdm.tqdm(os.listdir(self.list_path), desc="generating embeddings for holdout set..."):
            file_path = os.path.join(self.list_path, f)
            waveform, _ = librosa.load(file_path, sr=self.embedding_generator.processor_sr)
            self.audio_files[f] = self.embedding_generator.convert(waveform)

        self.num_holdout = len(self.audio_files)
        self.diameter = self._generate_diameter()


    def _generate_diameter(self):
        if not self.audio_files:
            raise Exception("audio files are not populated yet.")

        embeddings = list(self.audio_files.values())
        if len(embeddings) < 2:
            return 0.0

        pairwise_dists = pdist(embeddings, metric='euclidean')
        return np.max(pairwise_dists)

    def get_embedding(self, audio_file_path):
        """
        get the embedding of an audio file (path from root) based on our embedding generator
        """
        waveform, _ = librosa.load(audio_file_path, sr=self.embedding_generator.processor_sr)
        return self.embedding_generator.convert(waveform)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--list-path",
        type=str,
        required=True,
        help="path (from root) of directory holding only and all holdout set .wav files"
    )
    parser.add_argument(
        "-a", "--audio-path",
        type=str,
        help="path (from root) of audio file being examined (perhaps just for testing sake)"
    )

    parser.add_argument(
        "-d", "--directory-path",
        type=str,
        help="path (from root) of directory holding all audio files being examined (used to compute average metric)"
    )
    return parser

def main(args):


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if not os.path.isdir(args.list_path):
        raise Exception("please provide valid directory name")
    main(args)