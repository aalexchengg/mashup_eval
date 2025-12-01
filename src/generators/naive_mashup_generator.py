# Author: @echu2
from src.generators.base_mashup_generator import BaseMashupGenerator
import numpy as np
from typing import Tuple
import random
import logging
from typing import List, Dict
import os
import librosa
# import functions to load in audio
import sys
from src.utils import get_fma_paths, decode_audio

logger = logging.getLogger(__name__)

class NaiveMashupGenerator(BaseMashupGenerator):
    """
    Naively mashes up two songs by playing them over each other.
    """
    
    def generate(self, paths: List[str], out:str = None, layers: Dict[str, str] = None) -> Tuple[np.ndarray, int]:
        """
        Returns the average audio of both paths.
        @param paths: audio paths to mash songs up together.\\
        @param out: output filename of the generated song.\\
        @param layers: UNUSED in this function.\\
        @return audio, sr: the audio and sampling rate of the resulting mashup.
        """
        if len(paths) < 2:
            logger.error("Attempted to generate a mashup, but one or no filepaths were provided. Returning empty values")
            return np.empty(0), 0
        
        logger.info(f"Paths: {paths[0]} {paths[1]}")
        logger.info("Loading in audio of first path...")
        x1, sr = decode_audio(paths[0])
        logger.info("Loading in audio of second path...")
        x2, sr2 = decode_audio(paths[1])

        # resample if necessary
        if sr != sr2: 
            logger.info("Resampling...")
            x2 = librosa.resample(x2, orig_sr=sr2, target_sr=sr)

        if x1.shape[1] != x2.shape[1]: 
            logger.info(f"Padding as lengths of {x1.shape[1]} and {x2.shape[1]} are not the same")

            # pad if necessary
            max_len = max(x1.shape[1], x2.shape[1])
            x1 = librosa.util.fix_length(x1, size=max_len)
            x2 = librosa.util.fix_length(x2, size=max_len)

        x = x1 * 0.5 + x2 * 0.5
        if out:
            self._save_generation(x, sr, out)
        return x, sr


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logger.info("Testing loading and saving...")
    # Test that the generation works
    generator = NaiveMashupGenerator("test_mashup", "tmp")
    logger.info("Loading and sampling files...")
    # randomly sample two audio paths and create the generation
    paths = get_fma_paths(os.path.abspath('data')) # Replace with your own path
    samples = random.sample(paths, 2)
    logger.info("Generating and checking equality...")
    generator.generate(samples, "x1_copy")
    # assert that the generation is in fact the identity
    y, sr = decode_audio(f"{generator.out_dir}/x1_copy.wav")
    # clean up testing artifacts
    logger.info("Removing testing artifacts....")
    import os
    os.remove(f"{generator.out_dir}/x1_copy.wav")
    os.rmdir(generator.out_dir)
    # do the actual assertion
    logger.info("Asserting equality...")
    x1, sr1 = decode_audio(samples[0])
    assert(np.allclose(x1, y), 1e-4, 1e-4) # Might want to change tolerance? Passed the ear test.
    assert(sr1 == sr)
    logger.info("All tests passed.")


    

