# Author: @echu2
# Main file for evaluating mashups (or other audio). 
# Output metric is (k * d_HO(x)) / ((1-k) * (NLL(x)))

import argparse 
import csv
import os
import numpy as np
import tqdm
import yaml
from datetime import datetime
from src.evaluate.holdout_set import HoldoutSet
from src.evaluate.nll_extractor import NLLExtractor


import logging
logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio-path", 
        type=str, 
        required=True, 
        help="path for audio files to evaluate. note that this MUST be " \
        "either one wav file or a directory that contains wav files."
    )

    parser.add_argument(
        "-c", "--centroid", 
        type=bool, 
        # default=True, 
        help="whether our distance metric is based on centroid or nearest " \
        "neighbor. defaults to centroid."
    )

    parser.add_argument(
        "-o", "--output-path", 
        type=str, 
        # default=os.path.join("out", "eval_output"), 
        help="directory where output should be stored. " \
        "default is out/eval_output."
    )

    parser.add_argument(
        "-k", "--k",
        type=float, 
        # default=0.5, 
        help="k parameter for our metric. default is 0.5 (no effect of k)"
    )

    parser.add_argument(
        "-p", "--path-to-holdout", 
        type=str, 
        # default=os.path.join("data", "holdout_set"), 
        help="path (from root) of directory holding only and all" \
        " holdout set .wav files"
    )
    parser.add_argument('-config', type = str,
                        default = None,
                        help = "accepts yaml config files as well.")
    return parser

def metric(k, d_HO, NLL): 
    """
    main metric proposed. change this function for trying different metric
    equations.
    """
    return float(k * d_HO / ((1-k) * NLL))

def compute_dHO(H: HoldoutSet, audio_file_path, do_centroid): 
    return H.get_distance(audio_file_path=audio_file_path, 
                          is_centroid=do_centroid)
    
def compute_NLL(E: NLLExtractor, audio_file_path): 
    return E.get_nll(audio_file_path)

def main(args):
    k = args.k
    audio_path = args.audio_path
    do_centroid = args.centroid
    out_path = args.output_path
    holdout_path = args.path_to_holdout 
    logger.info(f"holdout path: {holdout_path}")

    # set up holdout & nll
    H = HoldoutSet(holdout_path)
    E = NLLExtractor()

    # set up output files
    current_time = datetime.now()
    timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M")

    all_metrics = []
    all_nlls = []
    out_csv = os.path.join(out_path, f"computed_values_{timestamp_string}.csv")
    csv_data = [["file_path", "is_centroid_distance", "d_HO", "NLL", "C_MU"]]
    out_txt = os.path.join(out_path, f"metric_summary_{timestamp_string}.txt")
    os.makedirs(out_path, exist_ok=True)
    num_audio_eval = 0

    # directory of audio files
    if os.path.isdir(audio_path): 
        for a in tqdm.tqdm(os.listdir(audio_path), desc="evaluating files..."): 
            file_path = os.path.join(audio_path, a)
            d_HO = compute_dHO(H, file_path, do_centroid)
            NLL = compute_NLL(E, file_path)
            C_MU = metric(k, d_HO, NLL)

            csv_data.append([file_path, do_centroid, d_HO, NLL, C_MU])
            all_metrics.append(C_MU)
            all_nlls.append(NLL)
            num_audio_eval += 1
        if num_audio_eval == 0: 
            raise Exception("no valid wav files were found in audio path. " \
            "please try again with a valid directory.")

    # one audio file
    elif (os.path.isfile(audio_path) 
          and os.path.splitext(audio_path)[1].lower() == ".wav"):
        d_HO = compute_dHO(H, audio_path, do_centroid)
        NLL = compute_NLL(E, audio_path)
        C_MU = metric(k, d_HO, NLL)

        csv_data.append([audio_path, do_centroid, d_HO, NLL, C_MU])
        all_metrics.append(C_MU)
        all_nlls.append(NLL)
        num_audio_eval += 1

    # incorrect input
    else: 
        raise Exception("input file was not wav or directory. please try again.")
        
    # save output to csv + txt
    with open(out_csv, 'w', newline='') as f: 
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    all_metrics = np.array(all_metrics)
    mean_metric, stdev_metric = np.mean(all_metrics), np.std(all_metrics)

    all_nlls = np.array(all_nlls)
    mean_nll, stdev_nll = np.mean(all_nlls), np.std(all_nlls)
    out_text = []
    out_text.append(f"Evaluation Metric: C_MU")
    out_text.append(f"Evaluation of {audio_path} using holdout set {holdout_path}" \
                    f", in total evaluating {num_audio_eval} files")
    out_text.append(f"Using d_HO as Euclidean distance " \
                    f"{'to centroid' if do_centroid else 'to nearest neighbor'}" \
                    " in holdout set")
    out_text.append(f"Mean C_MU: {mean_metric:.4f}")
    out_text.append(f"Standard Deviation C_MU: {stdev_metric:.4f}")
    out_text.append(f"Alternative Evaluation Metric of NLL (on MusicGen):")
    out_text.append(f"Mean NLL: {mean_nll:.4f}")
    out_text.append(f"Standard Deviation NLL: {stdev_nll:.4f}")
    
    with open(out_txt, 'w') as f: 
        f.writelines([line + "\n" for line in out_text])
        

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    # optional config override.
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Overwrite args with config values
        for key, value in config.items():
            setattr(args, key, value)
    main(args)
