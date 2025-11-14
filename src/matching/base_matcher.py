# Author: @abcheng. A base class for generating matches for mashing up.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict 
from matching.match import Match


class BaseMatcher(ABC):
    """
    A very empty base class that defines the function signature for a Matcher.
    """
    def __init__(self, out_dir :str = None):
        """
        Initializes, and creates an output directory if specified.\\
        @param out_dir: name of out_dir, if specified.
        """
        if out_dir:
            Path.mkdir(Path(out_dir), exist_ok= True)
        self.out_dir = out_dir

    @abstractmethod
    def generate_matches(self, sample_directory: str, max_size: int = -1, out_path:str = "match_out") -> List[Dict[int, Match]]:
        """
        Signature for generating matches based on a given directory of audio samples.\\
        @param sample_directory: filepath to where all the audio paths are.\\
        @param out: output directory of the result.\\
        @param max_size: maximum length of the generated matches.
        """
        pass

    @staticmethod
    def create(name, out_dir: str = None):
        """
        Factory method to create subclasses of BaseMatcher.\\
        Accepted mashers are "naive".\\
        @param name: name of the masher you want.\\
        @param out_dir: output directory of masher, if applicable\\
        """
        name = name.lower()
        if "naive" in name:
            from matching.naive_matcher import NaiveMatcher
            return NaiveMatcher(out_dir)
        else:
            raise ValueError(f"Unsupported matcher key: {name}")