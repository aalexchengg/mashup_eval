# Author @abcheng.
from matching.base_matcher import BaseMatcher
from matching.match import Match
from typing import List, Optional
from matching.cocola.contrastive_model import constants
from matching.cocola.contrastive_model.contrastive_model import CoCola
from matching.cocola.feature_extraction.feature_extraction import CoColaFeatureExtractor
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import uuid
import os
import sys
import logging

logger = logging.getLogger(__name__)

class CocolaMatcher(BaseMatcher):
    """
    A matcher that generates COCOLA scores to sort the output.
    """

    def _setup(self):
        """
        Adopted from https://github.com/gladia-research-group/cocola?tab=readme-ov-file.\\
        Creates a model loaded from a checkpoint as well as feature extractor.\\
        EmbeddingMode.BOTH means we are calculating scores for both harmonic and percussive.\\
        Other settings are EmbeddingMode.HARMONIC and EmbeddingMode.PERCUSSIVE.
        """
        # Load in the model.
        from matching.cocola import contrastive_model
        sys.modules['contrastive_model'] = contrastive_model
        model_path = os.path.abspath('../data/cocola_model/checkpoint-epoch=87-val_loss=0.00.ckpt')
        logger.info(f"Loading COCOLA model from {model_path}...")
        self.model = CoCola.load_from_checkpoint(model_path,
                                                 input_type=constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE)
        self.feature_extractor = CoColaFeatureExtractor()
        self.model.eval()
        self.model.set_embedding_mode(constants.EmbeddingMode.BOTH)
        logger.info("Finished loading model and feature extractor.")
        # Assert that stems exist.
        if self.stem_dir == None:
            raise AssertionError("Should have a valid stem directory. Run preprocess.py before proceeding.")
        # create a subdirectory for feature vectors.
        logger.info(f"Setting up feature vector sudirectory in {self.out_dir}")
        Path.mkdir(Path(f"{self.out_dir}/feature_vectors"), exist_ok= True)
    
    def _get_path(self, track_name, type, stored_data_path = "."):
        """
        This is copied over from automashup_utils.py for containment. Gets the path to the stem.
        """
        # returns the path of a song
        # It can return the whole track or just a separated part of it,
        # depending on the type, which should be one of the following :
        # 'entire', 'bass', 'drums', 'vocals', 'other'
        # Extract the filename without extension
        track_name_no_ext = os.path.splitext(track_name)[0]

        if type == 'entire':
            if os.path.exists(f'{stored_data_path}/input/{track_name}'):
                path = f'{stored_data_path}/input/{track_name}'
            else:
                path = f'{stored_data_path}/input/{track_name_no_ext}.wav' # Check for .wav if .mp3 is not found
                if not os.path.exists(path):
                    path = f'{stored_data_path}/input/{track_name_no_ext}.mp3' # Check for .mp3 if .wav is not found
        else:
            path = f'{stored_data_path}/separated/htdemucs/{track_name_no_ext}/{type}.wav'
            if not os.path.exists(path):
                path = f'{stored_data_path}/separated/htdemucs/{track_name_no_ext}/{type}.mp3'
                if not os.path.exists(path): # Check for common extension mismatches
                    path = f'{stored_data_path}/separated/htdemucs/{track_name_no_ext}/{type}.wav'
                    if not os.path.exists(path):
                        path = f'{stored_data_path}/separated/htdemucs/{track_name_no_ext}/{type}.mp3'
        # assert(os.path.exists(path)), f"File not found: {path}" # Added a more informative error message
        if not os.path.exists(path):
            return None
        return path

    def _get_features(self, song: str, is_instrumental:bool = False, out:str = None) -> Optional[torch.Tensor]:
        """
        Gets the features of a song using the COCOLA Feature extractor.\\
        @param song: the song we want to extract features from.\\
        @param is_instrumental: whether we are extracting vocal or instrumental features.\\
        @param out: the output path of the resulting feature tensor.\\
        @returns None if we can't extract the feature for the specified layer, otherwise the features of the layer(s).
        """
        # if we've already gotten the features return immediately
        logger.info(f"Generating features for {song}. Instrumental status: {is_instrumental}.")
        if out and os.path.exists(out):
            logger.info("Feature vector already exists. Retrieving...")
            features = torch.load(out)
            return features
        # check which layers we want to extract from
        types = self.instrumental if is_instrumental else ['vocals']
        logger.info("Generating waveform...")
        result_waveform = None 
        for type in types:
            # if the layer exists...
            if self._get_path(song, type, self.stem_dir) != None:
                path = self._get_path(song, type, self.stem_dir)
                waveform, base_sr = torchaudio.load(path)
                # add it to the current waveform
                result_waveform = waveform if result_waveform is None else result_waveform[:waveform.shape[0]] + waveform[:result_waveform.shape[0]]
        if result_waveform == None:
            return None
        # extract the features of the resulting waveform!
        logger.info("Extracting features...")
        features = self.feature_extractor(result_waveform)
        # save if needed
        if out:
            logger.info(f"Saving feature vector to {out}...")
            torch.save(features, out)
        return features
    
    def _build_match(self, vocal_song: str, instrumental_song: str, sample_directory: str) -> Optional[Match]:
        """
        For a given song for vocals and song for instrumentals, build a Match object and calculate its score.\\
        @param vocal_song: the song name that we want to extract the vocals from.\\
        @param instrumental_song: the song name that we want to extract the instrumentals from.\\
        @param sample_directory: where the songs (NOT the stems) live.\\
        @returns a Match object if a match is feasible, otherwise None.
        """
        # first, attempt to get the features.
        logger.info("Begin match building.")
        logger.info("Extracting features...")
        vocal_out = f"{self.out_dir}/feature_vectors/{vocal_song}_features.pt"
        instrumental_out  = f"{self.out_dir}/feature_vectors/{instrumental_song}_features.pt"
        vocal_features = self._get_features(vocal_song, False, vocal_out)
        instrumental_features = self._get_features(instrumental_song, True, instrumental_out)
        # then, if both features exist....
        if vocal_features != None and instrumental_features != None:
            logger.info("Match is possible. extracting score...")
            # get the max length for truncation (lengths must match + computational complexity)
            max_length = min(vocal_features.shape[-1], instrumental_features.shape[-1], 100) # TODO: smarter way to set max_length? If its too large, it kills the program.
            # calculate the harmonic and percussive score, and get the average
            composite_score = self.model.score(vocal_features[..., :max_length], instrumental_features[..., :max_length])
            score = composite_score.mean().item()
            # finally, construct the match object and return it.
            layers = {"vocals": vocal_song, "bass": instrumental_song, "drums": instrumental_song, "other": instrumental_song}
            match = Match(uuid.uuid4(), 
                              sample_directory, 
                              [vocal_song, instrumental_song],
                              score,
                              layers)
            return match
        # if either feature doesnt exist, return None.
        logger.info("Match is not possible because one of the vocal or instrumental does not exist. Returning None.")
        return None


    def generate_matches(self, sample_directory: str, max_size: int=-1, out_path: str="match_out", sort: str="unsorted") -> List[Match]:
        """
        Generate song matches with corresponding COCOLA scores.\\
        @param sample_directory: the directory where the songs live.\\
        @param max_size: maximum size of the resulting generation.\\
        @param out_path: the output path of the jsonl in its given directory.\\
        @param sort: the sorting order wanted. options are ["unsorted", "largest", "smallest"]\\
        @returns a list of Match objects, sorted based on the sort argument.
        """
        result = []
        all_songs = []
        for entry_name in os.listdir(sample_directory):
            if entry_name.split(".")[-1] == "mp3": # ensures that file suffix is an mp3
                all_songs.append(entry_name)
        # create pairwise entries with no repeats.
        for i in range (len(all_songs)):
            for j in range(i+1, len(all_songs)):
                # for each possible pair of songs, try song i as vocal and song j as instrumental
                logger.info(f"Generating match with {all_songs[i]} as vocal and {all_songs[j]} as instrumental...")
                match1 = self._build_match(all_songs[i], all_songs[j], sample_directory)
                # then, try song i as instrumental and song j as vocal
                logger.info(f"Generating match with {all_songs[j]} as vocal and {all_songs[i]} as instrumental...")
                match2 = self._build_match(all_songs[j], all_songs[i], sample_directory)
                # add to list of match is valid.
                if match1 != None:
                    result.append(match1)
                if match2 != None:
                    result.append(match2)
        # truncate if necessary
        if max_size > 0:
            logger.info(f"Truncating to size {max_size}...")
            result = result[:max_size]
        if sort == "largest":
            logger.info("Sorting with largest score first...")
            result.sort(reverse=True)
        elif sort == "smallest":
            logger.info("Sorting with smallest score first...")
            result.sort()
        # write to out path
        if self.out_dir:
            out_path = f"{self.out_dir}/{out_path}"
        logger.info(f"Writing to {out_path}.jsonl...")
        with open(f"{out_path}.jsonl", "w") as file:
            for item in result:
                file.write(item.to_json() + '\n')
        # and also return the result
        return result