# Author: @abcheng. Adopted from https://github.com/ax-le/automashup/blob/main/automashup/Notebooks/Tutorial%20notebook%20-%20make%20a%20mashup.ipynb.
from generators.base_mashup_generator import BaseMashupGenerator
import numpy as np
from typing import Tuple
import random
import logging
from typing import List

from automashup.track import Track
from automashup.automashup_utils import key_finder
from automashup import automashup as mashupper
import random

logger = logging.getLogger(__name__)

class AutoMashupGenerator(BaseMashupGenerator):

    def setup(self):
        """
        Creates a preprocessing directory, so we don't have to re-preprocess each one on the fly.
        """
        self.preprocess_dir = self.create_out_dir('/Users/abcheng/Documents/workspace/mashup_eval/data/auto_preprocess') # set as absolute path
        self.layers = ['vocals', 'bass', 'drums', 'other']
    
    
    def generate(self, paths: List[str], out = None) -> Tuple[np.ndarray, int]:
        """
        Generates a song based on the audio paths. 
        @param paths: audio paths to songs we want to mash up
        @param out: optional output path, without the ext
        @return y, sr: the audio and the sampling rate of the resulting mashup
        """
        songs = []
        for path in paths:
            # preprocessing should have already occurred.
            key_finder(path, stored_data_path=self.preprocess_dir)
            song_with_ext = path.split("/")[-1] # extracts out foo.mp3
            songs.append(song_with_ext) # foo.mp3 --> foo
        # we're going to randomly choose from the paths we have and generate our mashups
        tracks = []
        for layer in self.layers:
            candidate = random.choice(songs) # TODO: there's a chance we generate the same song for all candidates. can we prevent that?
            candidate_track = Track.track_from_song(candidate, type=layer, stored_data_path=self.preprocess_dir)
            tracks.append(candidate_track)
        # now, we want to apply the mashup function
        result = mashupper.mashup_technic_fit_phase_repitch(tracks)
        # save output if watned.
        if out:
            self.save_generation(result.audio, result.sr, out)
        return result.audio, result.sr

if __name__ == "__main__":
    generator = AutoMashupGenerator("test run")
    paths = ['/Users/abcheng/Documents/workspace/mashup_eval/data/sample/Bam Bam - Hi-Q.mp3',
             '/Users/abcheng/Documents/workspace/mashup_eval/data/sample/ZOE.LEELA - Jewel.mp3']
    generator.generate(paths, "attempt")

