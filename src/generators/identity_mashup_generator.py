# Author: @abcheng
from generators.base_mashup_generator import BaseMashupGenerator
from dataclasses import dataclass
import numpy as np
from typing import Tuple
import random
import logging
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class IdentityMashupGenerator(BaseMashupGenerator):
    """
    Trivially returns the first mashup and ignores the second mashup.
    """
    
    def generate(self, paths: List[str], out = None) -> Tuple[np.ndarray, int]:
        """
        Returns the first audio from the given paths.
        @param paths: audio paths to mash songs up together
        @param out: output filename of
        """
        if len(paths) == 0:
            logger.error("Attempted to generate a mashup, but no filepaths were provided. Returning empty values")
            return np.empty(0), 0
        x1, sr1 = decode_audio(paths[0])
        if out:
            self.save_generation(x1, sr1, out)
        return x1, sr1


if __name__ == "__main__":
    # import functions to load in audio
    import sys
    sys.path.append("..")
    from utils import get_fma_paths, decode_audio
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logger.info("Testing loading and saving...")
    # Test that the generation works
    generator = IdentityMashupGenerator("test_mashup", "tmp")
    logger.info("Loading and sampling files...")
    # randomly sample two audio paths and create the generation
    paths = get_fma_paths("/Users/abcheng/Documents/workspace/mashup_eval/data") # Replace with your own path
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


    

