# Author: @abcheng. Written for 15-798 F25 Final Project. Some code taken from assignment 1.
import pathlib
import logging
from typing import Optional, Tuple, List
import librosa
import numpy as np
import functools
import random
from tqdm import tqdm
import math
logger = logging.getLogger("utils.py")


def get_fma_dir(root: str = "") -> pathlib.Path:
    """
    Gets the directory to the FMA dataset.
    @param root: the parent directory of the FMA dataset
    @returns: Path to FMA dataset
    """
    fma_dir = pathlib.Path(f"{root}/fma_small")
    if not fma_dir.is_dir():
        raise FileNotFoundError(f"FMA data directory not found.")
    return fma_dir

def get_fma_paths(root: str = "") -> List[str]:
    """
    Gets a sorted list of all the FMA mp3 files.
    @param root: the parent directory fo the FMA dataset.
    @returns: List of filepaths
    """
    fma_mp3_paths = sorted(list(get_fma_dir(root).glob('**/*.mp3')))
    return fma_mp3_paths

@functools.lru_cache(maxsize=16)
def decode_audio(path: str, offset: float = 0.0, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Gets the numpy array and sampling rate of a FMA audio example. 
    Added caching in case we need to load same audio many times.
    @param path: the path to the specific example (e.g. "fma_small/000/000002.mp3")
    @param offset: start reading after this time (in seconds)
    @param duration: only load up to this much audio (in seconds)
    @returns: audio time series, sampling rate
    """
    x, sr = librosa.load(path, sr=None, mono=False, offset=offset, duration=duration)
    if x.ndim == 1:
        x = x[np.newaxis]
    return x, sr

# From: https://github.com/ax-le/automashup/blob/main/automashup/src/utils.py
def note_to_frequency(key):
    # turn a note with a mode to a frequency
    note, mode = key.split(' ', 1)
    reference_frequency=440.0
    semitone_offsets = {'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6, 'E': -5, 'Fb': -5, 'E#': -4,
                        'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1, 'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'B#': 3}
    semitone_offset = semitone_offsets[note]
    if mode == 'minor':
        semitone_offset -= 3
    frequency = reference_frequency * 2 ** (semitone_offset / 12)
    return frequency

# From: https://github.com/ax-le/automashup/blob/main/automashup/src/utils.py
def calculate_pitch_shift(source_freq, target_freq):
    pitch_shift = 12 * math.log2(target_freq / source_freq)
    return pitch_shift

# From: https://github.com/ax-le/automashup/blob/main/automashup/src/utils.py
def increase_array_size(arr, new_size):
    if len(arr) < new_size:
        # Create a new array with the new size
        increased_arr = np.zeros(new_size)
        # Copy elements from the original array to the new array
        increased_arr[:len(arr)] = arr
        return increased_arr
    else:
        return arr

# From: https://github.com/ax-le/automashup/blob/main/automashup/src/utils.py
def closest_index(value, value_list):
    # get the index of the closest value of a specific target in a list
    closest_index = min(range(len(value_list)), key=lambda i: abs(value_list[i] - value))
    return closest_index

if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    # sanity check that all the util functions work
    logger.debug("Checking path loading works...")
    paths = get_fma_paths("..")
    logger.info(f"FMA has {len(paths)} samples")
    samples = random.sample(paths, 3)
    logger.debug("Checking sample loading...")
    for sample in tqdm(samples):
        arr, sr = decode_audio(sample, duration = 10)
        assert((arr.shape[-1] / sr )== 10)
    logger.debug('All tests passed.')
