# Author: @echu2
# Main file for running metric suite

import argparse
from functools import cache
import librosa
import tqdm
import numpy as np
import os
from src.evaluate.embedding_generator import EmbeddingGenerator
from scipy.spatial.distance import pdist

def dist(a, b): 
    """
    return euclidean distance between two m x n arrays.
    """
    if a.shape[0] != b.shape[0]: 
        # take minimum timestep between a and b
        time_steps = min(a.shape[0], b.shape[0])
        a, b = a[:time_steps, ...], b[:time_steps, ...]
    elif a.shape[1] != b.shape[1]: 
        raise Exception("embeddings of different dimensions are not supported.")
    return np.linalg.norm(a - b)

class HoldoutSet:

    def __init__(self, list_path, embed_model_name=None):
        self.list_path = list_path
        self.audio_embeddings = dict()
        if embed_model_name:
            self.embedding_generator = EmbeddingGenerator(embed_model_name)
        else:
            self.embedding_generator = EmbeddingGenerator()

        self.min_timesteps = 10 ** 6
        for f in tqdm.tqdm(os.listdir(self.list_path), 
                           desc="generating embeddings for holdout set..."):
            file_path = os.path.join(self.list_path, f)
            self.audio_embeddings[f] = self.get_embedding(file_path)
            self.min_timesteps = min(self.min_timesteps, 
                                     self.audio_embeddings[f].shape[0])

        self.num_holdout = len(self.audio_embeddings)
        # truncate
        for f in self.audio_embeddings.keys(): 
            self.audio_embeddings[f] = self.audio_embeddings[f][:self.min_timesteps, ...]
        self.diameter = self._generate_diameter()
        self.centroid = self._find_centroid()


    def _generate_diameter(self):
        """
        generates diameter of all embeddings of holdout set.
        this might not be necessary for the evaluation metric proposed.
        NOTE: this is currently commented out because it takes forever, 
        even for the 80 mini-example... if it's not necessary anyway
        might as well as not compute it
        """
        if not self.audio_embeddings:
            raise Exception("audio files are not populated yet.")

        embeddings = list(self.audio_embeddings.values())
        if len(embeddings) < 2:
            return 0.0

        pairwise_dists = []
        for i in tqdm.tqdm(range(len(embeddings)-1), desc="getting pairwise dists..."): 
            for j in range(i+1, len(embeddings)): 
                pairwise_dists.append(dist(embeddings[i], embeddings[j]))
        return max(pairwise_dists)
    
    def _find_centroid(self): 
        """
        finds the center-most point of the FMA dataset in embedding space. 
        potentially used for evaluation metric (as a measure of "uniqueness")
        """
        if not self.audio_embeddings: 
            raise Exception("audio files are not populated yet.")
        
        embeddings = np.array(list(self.audio_embeddings.values()))
        return np.mean(embeddings, axis=0)

    @cache
    def get_embedding(self, audio_file_path):
        """
        get the embedding of an audio file (path from root) based on our embedding generator
        """
        # theoretically if audio_file_path has been seen before this is cached
        waveform, _ = librosa.load(audio_file_path, sr=self.embedding_generator.processor_sr)
        return self.embedding_generator.convert(waveform)
    
    def distance_to_centroid(self, audio_file_path): 
        embedding = self.get_embedding(audio_file_path)
        return dist(embedding, self.centroid)

    def distance_to_nearest_neighbor(self, audio_file_path): 
        min_dist = 10 ** 6
        min_dist_file = None
        embedding = self.get_embedding(audio_file_path).numpy()
        for k, v in self.audio_embeddings.items(): 
            d = dist(embedding, v.numpy())
            if d < min_dist: 
                min_dist = d
                min_dist_file = k
        
        return min_dist, min_dist_file

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--list-path",
        type=str,
        default=os.path.join("data", "holdout_set"),
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
    holdout = HoldoutSet(args.list_path)
    if args.audio_path: 
        print(f"calculating metric for audio at {args.audio_path}...")

        cache_info_1 = holdout.get_embedding.cache_info()
        print(f"distance to centroid: {holdout.distance_to_centroid(args.audio_path)}")

        cache_info_2 = holdout.get_embedding.cache_info()
        min_dist, min_dist_file = holdout.distance_to_nearest_neighbor(args.audio_path)
        print(f"distance to nearest neighbor: {min_dist} is file {min_dist_file}")

        cache_info_3 = holdout.get_embedding.cache_info()

        if holdout.diameter: 
            print(f"diameter of holdout set: {holdout.diameter}")

        print(f"cache info before calling once: {cache_info_1}")
        print(f"cache info after calling once: {cache_info_2}")
        print(f"cache info after calling twice: {cache_info_3}")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if not os.path.isdir(args.list_path):
        raise Exception("please provide valid directory name")
    main(args)

