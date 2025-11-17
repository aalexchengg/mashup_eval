# Author @abcheng. Adopted from https://github.com/ax-le/automashup/blob/main/automashup/src/utils.py.
import numpy as np
import math
import os
import json
from generators.automashup.key_finder import KeyFinder

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

def extract_filename(file_path):
    # Extract filename from a given path
    filename = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(filename)
    return filename_without_extension

def key_finder(path, stored_data_path="."):
    filename = extract_filename(path)
    struct_path = f"{stored_data_path}/struct/{filename}.json"
    with open(struct_path, 'r') as file:
        data = json.load(file)
        data['key'] = KeyFinder(path).key_dict
    with open(struct_path, 'w') as file:
        json.dump(data, file, indent=2)


def get_path(track_name, type, stored_data_path = "."):
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