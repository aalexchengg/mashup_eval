# Author: @abcheng
from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import os
from pathlib import Path
import logging
import soundfile as sf

logger = logging.getLogger(__name__)

class BaseMashupGenerator:
    """
    Abstract class for a MashupGenerator. Should be inherited to be any use.
    """
    def __init__(self, name: str, out_dir: str = None):
        """
        Initializes a generator with a given name
        @param name: the name of the mashup generator
        @param generate_out_dir: whether to generate the output directory for this mashup generator
        """
        self.name = name
        self.out_dir = self.create_out_dir(out_dir)
        self.setup()
    
    def setup(self):
        """
        Optional method to do post initialization.
        """
        pass
    
    def __str__(self):
        """
        print(generator) should return generator.name
        """
        return self.name 
    
    def create_out_dir(self, out_dir: str) -> str:
        """
        Creates an output directory. If nothing is passed, creates a default output directory.
        @param out_dir: name of output directory, if exists.
        @returns resulting name of output directory.
        """
        if out_dir == None:
            out_dir = f"{self.name}_out"
            logger.info(f"No output directory passed. setting it as {out_dir}")
        Path.mkdir(Path(out_dir), exist_ok= True)
        return out_dir

        
    def save_generation(self, x: np.ndarray, sr: int, filename: str) -> None:
        """
        Saves a generated audio to a filename.
        @param x: np array representing the audio
        @param sr: sampling rate of the audio
        @param filename: filename to save it to
        """
        # Based on: https://stackoverflow.com/questions/73239578/couldnt-store-audio-as-mp3-file-using-soundfile
        # Since FMA is a mp3 dataset, so we need to transpose.
        sf.write(f"{self.out_dir}/{filename}.wav", np.transpose(x), sr)


    @abstractmethod
    def generate(paths: List[str], out = None) -> Tuple[np.ndarray, int]:
        """
        Generates a mashup.
        @param paths: list of filepaths of songs we will be mashing up. We assume they are all mp3 for now
        @param out: the output directory of the mashup, if specified
        @returns: the numpy array and the sampling rate of the resulting mashup 
        """
        pass

    @staticmethod
    def create(name: str, out_dir: str = None):
        """
        Factory method to create subclasses of BaseMashupGenerators.
        Accepted generators are "identity" and "auto".
        @param name: name of the generator
        @param out_dir: output directory, if specified
        @returns a child BaseMashupGenerator
        """
        name = name.lower()
        if "identity" in name:
            from mashup_eval.src.generators.identity_mashup_generator import IdentityMashupGenerator
            return IdentityMashupGenerator(name, out_dir)
        elif "auto" in name:
            from mashup_eval.src.generators.auto_mashup_generator import AutoMashupGenerator
            return AutoMashupGenerator(name, out_dir)
        else:
            raise ValueError(f"Unsupported generator key: {name}")
    