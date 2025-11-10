# Author: @abcheng
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseMashupGenerator:
    """
    Abstract class for a MashupGenerator. Should be inherited to be any use.
    """
    def __init__(self, name: str):
        """
        Initializes a generator with a given name
        """
        self.name = name
    
    @abstractmethod
    def generate(x1 : np.npdarray , sr1 : int, x2 : np.ndarray, sr2 : int, out = None) -> Tuple[np.ndarray, int]:
        """
        Generates a mashup.
        @param x1: the np array of the first song
        @param sr1: the sampling rate of the first song
        @param x2: the np array of the second song
        @param sr2: the sampling rate of the second song
        @param out: the output directory of the mashup, if specified
        @returns: the numpy array and the sampling rate of the resulting mashup 
        """
        pass

    def __str__(self):
        """
        print(generator) should return generator.name
        """
        return self.name 
