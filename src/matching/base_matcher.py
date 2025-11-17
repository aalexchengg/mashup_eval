# Author: @abcheng. A base class for generating matches for mashing up.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict 
from matching.match import Match
import logging 

logger = logging.getLogger(__name__)

class BaseMatcher(ABC):
    """
    A very empty base class that defines the function signature for a Matcher.
    """
    def __init__(self, name, out_dir :str = None, stem_dir: str = None):
        """
        Initializes, and creates an output directory if specified.\\
        @param out_dir: name of out_dir, if specified.
        """
        self.name = name
        self.out_dir = self._create_out_dir(out_dir)
        self.stem_dir = stem_dir
        print(f"stem dir is {self.stem_dir} {stem_dir}")
        self.instrumental = ['bass', 'drums', 'other']
        self.setup()

    def _create_out_dir(self, out_dir: str) -> str:
        """
        Creates an output directory. If nothing is passed, creates a default output directory.\\
        @param out_dir: name of output directory, if exists.\\
        @returns resulting name of output directory.
        """
        if out_dir == None:
            out_dir = f"{self.name}_out"
            logger.info(f"No output directory passed. setting it as {out_dir}")
        Path.mkdir(Path(out_dir), exist_ok= True)
        return out_dir
    
    def setup(self):
        # optional setup method to override.
        pass

    @abstractmethod
    def generate_matches(self, sample_directory: str, max_size: int = -1, out_path:str = "match_out", sort = "unsorted") -> List[Match]:
        """
        Signature for generating matches based on a given directory of audio samples.\\
        @param sample_directory: filepath to where all the audio paths are.\\
        @param out: output directory of the result.\\
        @param max_size: maximum length of the generated matches.\\
        @returns a list of matches.
        """
        pass

    @staticmethod
    def create(name, out_dir: str = None, stem_dir: str = None):
        """
        Factory method to create subclasses of BaseMatcher.\\
        Accepted mashers are ["naive", "cocola"].\\
        @param name: name of the masher you want.\\
        @param out_dir: output directory of masher, if applicable.\\
        @returns a Matcher child class.
        """
        name = name.lower()
        if "naive" in name:
            from matching.naive_matcher import NaiveMatcher
            return NaiveMatcher(name, out_dir)
        elif "cocola" in name:
            from matching.cocola_matcher import CocolaMatcher
            return CocolaMatcher(name, out_dir, stem_dir)
        else:
            raise ValueError(f"Unsupported matcher key: {name}")