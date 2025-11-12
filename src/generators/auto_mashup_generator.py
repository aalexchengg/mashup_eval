# Author: @abcheng
from base_mashup_generator import BaseMashupGenerator
from dataclasses import dataclass
import numpy as np
from typing import Tuple
import random
import logging
from typing import List

from automashup.track import Track
from automashup import automashup as mashupper
import allin1
import warnings
import random

logger = logging.getLogger(__name__)

@dataclass
class AutoMashupGenerator(BaseMashupGenerator):
    def __post_init__(self):
        """
        Creates a preprocessing directory, so we don't have to re-preprocess each one on the fly.
        """
        self.preprocess_dir = self.create_out_dir('auto_preprocess')
        self.layers = ['vocals', 'bass', 'drums', 'other']
    
    
    def generate(self, paths: List[str], out = None) -> Tuple[np.ndarray, int]:
        songs = []
        for path in paths:
            # check if we have already preprocessed it. if not, do it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Actually doesn't work...
                allin1.analyze(path, out_dir=f'{self.preprocess_dir}/struct', demix_dir=f'{self.preprocess_dir}/separated', keep_byproducts=True, overwrite=False)
            # TODO: build or find a keyfinder
            # key_finder(song_path, stored_data_path=stored_data_path)
            songs.append(path.split(".")[0]) # foo.mp3 --> foo
        # we're going to randomly choose from the paths we have and generate our mashups
        tracks = []
        for layer in self.layers:
            candidate = random.choice(songs)
            candidate_track = Track.track_from_song(candidate, type=layer, stored_data_path=self.preprocess_dir)
            tracks.append(candidate_track)
        # now, we want to apply the mashup function
        result = mashupper.mashup_technic_fit_phase_repitch(tracks)
        if out:
            self.save_generation(result.audio, result.sr, out)
        return result.audio, result.sr

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils import get_fma_paths, decode_audio
    generator = AutoMashupGenerator("test run")
    paths = ['/Users/abcheng/Documents/workspace/mashup_eval/sample/Play House - Afro Time.mp3',
             '/Users/abcheng/Documents/workspace/mashup_eval/sample/Rabato - ass slap.mp3']
    generator.generate(paths)

