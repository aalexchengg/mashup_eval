# Author: @echu2
# Assumes FMA data is already downloaded in root folder

import argparse
import json
import os
import soundfile as sf
import numpy as np
import pydub 
import tqdm

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--fma_path", 
        type=str, 
        default=os.path.join("data", "fma_small"),
        help="path of FMA folder (likely data/fma_small)", 
    )
    parser.add_argument(
        "-p", "--percent", 
        type=float,
        default=0.1, 
        help="percent of entire FMA folder to select for holdout set"
    )
    parser.add_argument(
        "-s", "--save_path", 
        type=str, 
        default=os.path.join("data", "holdout_set"),
        help="folder to save holdout set"
    )
    return parser


def main(args): 
    np.random.seed(798)

    fma_path = args.fma_path
    p = args.percent
    out_path = args.save_path

    # gather list of all mp3 file names
    file_names = []

    for root, dirs, files in os.walk(fma_path):
        file_names.extend([os.path.join(root, f) for f in files])

    # randomly select 10
    selected_idxs = np.random.choice(range(len(file_names)), 
                                     size=int(len(file_names) * p), 
                                     replace=False)

    # save them as new wav files in out folder
    os.makedirs(out_path, exist_ok=True)
    final_file_names = []

    # save to out folder + json
    for i in tqdm.tqdm(selected_idxs, desc="converting mp3 into wav..."): 
        file_name = file_names[i]
        try: 
            pydub_file = pydub.AudioSegment.from_mp3(file_name)
            file_as_np = np.array(pydub_file.get_array_of_samples())
            if pydub_file.channels == 2: 
                file_as_np = file_as_np.reshape((-1, 2))
            new_file_name = os.path.splitext(os.path.split(file_name)[1])[0]
            new_file_name += ".wav"
            new_file_name = os.path.join(out_path, new_file_name)
            
            pydub_file.export(new_file_name, format="wav")
            final_file_names.append(new_file_name)
        except: 
            print(f"could not write file {file_name}, skipping")
            continue

    # out_json = os.path.join(out_path, "file_names.json")
    # with open(out_json, 'w', encoding='utf-8') as f: 
    #     json.dump(final_file_names, f, indent=4)

if __name__ == "__main__": 
    parser = setup_parser()
    args = parser.parse_args()
    if not args.fma_path or not os.path.isdir(args.fma_path): 
        raise Exception("please provide valid FMA path")
    main(args)