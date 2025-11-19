# Author: @echu2
# File for computing holdout set metric (d_HO)

import argparse
from functools import cache
import librosa
import tqdm
import torch
import numpy as np
import os
import math
from transformers import AutoModel, Wav2Vec2FeatureExtractor

def dist(a, b):
    """
    euclidean distance between two time-aligned embeddings (T x D).
    if T differs, truncate both to min(T).
    """
    t = min(a.shape[0], b.shape[0])
    if a.shape[1] != b.shape[1]:
        raise ValueError("Embedding dimensions must match.")
    return np.linalg.norm(a[:t] - b[:t])

class EmbeddingGenerator:
    """
    internal class used for:
      - loading model and processor
      - loading audio
      - converting audio into a final-layer embedding
      - returning numpy arrays
    """
    def __init__(self, model_name="m-a-p/MERT-v1-95M"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor_sr = self.processor.sampling_rate

    def load_audio(self, audio_file_path):
        waveform, _ = librosa.load(audio_file_path, sr=self.processor_sr)
        return waveform

    def _convert_mert(self, input_audio):
        inputs = self.processor(
            input_audio,
            sampling_rate=self.processor_sr,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return torch.stack(outputs.hidden_states).squeeze()

    def convert(self, input_audio):
        """
        modular convert from audio to embeddings
        """
        if "MERT" in self.model_name:
            full_stack = self._convert_mert(input_audio)
            last_layer = full_stack[-1]
            return last_layer.cpu().numpy()
        else:
            raise NotImplementedError("Only MERT-like models supported.")

class HoldoutSet:
    """
    client class to access embeddings, used for:
        - caching of embeddings
        - holdout set centroid
        - nearest neighbor metric
        - automatic slicing for long embeddings (with delta threshold)
        - supports input shorter than holdout via dist() truncation
    """

    def __init__(self, list_path, embed_model_name=None, delta=5):
        self.list_path = list_path
        self.delta = delta # threshold for slicing
        self.embedding_generator = EmbeddingGenerator(
            embed_model_name if embed_model_name else "m-a-p/MERT-v1-95M"
        )

        self.audio_embeddings = {}
        self.min_timesteps = float("inf")
        for fname in tqdm.tqdm(os.listdir(self.list_path),
                               desc="Generating embeddings for holdout set..."):
            file_path = os.path.join(self.list_path, fname)
            emb = self.get_embedding(file_path)
            self.audio_embeddings[fname] = emb
            self.min_timesteps = min(self.min_timesteps, emb.shape[0])

        for f in self.audio_embeddings:
            self.audio_embeddings[f] = self.audio_embeddings[f][:self.min_timesteps]

        self.num_holdout = len(self.audio_embeddings)
        self.centroid = self._find_centroid()

    @cache
    def get_embedding(self, audio_file_path):
        """
        cached: load audio + convert to embedding.
        returns numpy array (T x D).
        """
        waveform = self.embedding_generator.load_audio(audio_file_path)
        return self.embedding_generator.convert(waveform)

    def _find_centroid(self):
        emb_stack = np.stack(list(self.audio_embeddings.values()))
        return np.mean(emb_stack, axis=0)

    def _slice_embedding(self, emb):
        """
        if emb is longer than the holdout minimum length by more than delta,
        return a list of evenly spaced slices of length min_timesteps.
        Otherwise:
            - If T <= M: return [emb]  (handled naturally)
            - If M < T <= M + delta: truncate once [emb[:M]]
        """
        T = emb.shape[0]
        M = self.min_timesteps
        if T <= M:
            return [emb]
        
        if T - M <= self.delta:
            return [emb[:M]]

        n = math.ceil(T / M)
        stride = (T - M) // (n - 1)

        slices = []
        for i in range(n):
            start = i * stride
            end = start + M
            slices.append(emb[start:end])

        return slices
    
    def get_distance(self, audio_file_path, is_centroid): 
        if is_centroid: 
            return self.distance_to_centroid(audio_file_path)
        else: 
            return self.distance_to_nearest_neighbor(audio_file_path)
        
    def distance_to_centroid(self, audio_file_path):
        emb = self.get_embedding(audio_file_path)
        windows = self._slice_embedding(emb)
        dists = [dist(w, self.centroid) for w in windows]
        return float(np.mean(dists))

    def distance_to_nearest_neighbor(self, audio_file_path):
        emb = self.get_embedding(audio_file_path)
        windows = self._slice_embedding(emb)

        all_dists = []
        for w in windows:
            min_d = float("inf")
            for hold_emb in self.audio_embeddings.values():
                d = dist(w, hold_emb)
                if d < min_d:
                    min_d = d
            all_dists.append(min_d)

        return float(np.mean(all_dists))

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--list-path",
        type=str,
        default=os.path.join("data", "holdout_set"),
        help="Directory containing holdout .wav files"
    )
    parser.add_argument(
        "-a", "--audio-path",
        type=str,
        help="Audio file to evaluate"
    )
    return parser

def main(args):
    holdout = HoldoutSet(args.list_path)

    if args.audio_path:
        print(f"Evaluating: {args.audio_path}")

        print("Distance to centroid:")
        print(holdout.distance_to_centroid(args.audio_path))

        print("Nearest-neighbor distance:")
        print(holdout.distance_to_nearest_neighbor(args.audio_path))


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if not os.path.isdir(args.list_path):
        raise Exception("Invalid holdout directory.")
    main(args)

