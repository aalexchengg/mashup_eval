# @author abcheng. used to create the survey.
import pandas as pd
from typing import Optional
from tqdm import tqdm
import numpy as np
import json
import argparse
import logging
import shutil

logger = logging.getLogger(__name__)

# TODO: make this an argparse
# evaluate filepaths
naive_eval_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/naive_eval_output.csv"
auto_eval_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/auto_eval_output.csv"

# match filepaths
naive_match_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/psc_naive_match.jsonl"
auto_match_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/cocola_out/full_match_out.jsonl"

auto_out_dir = "/Users/abcheng/Documents/workspace/mashup_eval/out/auto_out"
naive_out_dir = "/Users/abcheng/Documents/workspace/mashup_eval/out/psc_naive_mashups"
THRESHOLD = 5

def extract_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the id from a filepath and replaces the column.\\
    @param df: the dataframe to operate on.\\
    @returns a dataframe with a new id variable.
    """
    df['id'] = df['file_path'].apply(lambda x: x.split("/")[-1].split(".")[0])
    del df['file_path'] # save memory
    return df


def load_df(eval_filepath: str, match_filepath: Optional[str]) -> pd.DataFrame:
    """
    Loads in a dataframe and merges it with its match scores.\\
    @param eval_filepath: path to the evaluation dataframe.\\
    @param match_filepath: path to the match information, if available.\\
    @returns dataframe with necessary information.
    """
    df = pd.read_csv(eval_filepath)
    df = extract_id(df)
    # should also load in the match scores, if available
    if match_filepath:
        with open(match_filepath, 'r') as f:
            matches = [json.loads(line) for line in f]
        # need to coerce into a string first to remove nans
        for match in matches:
            match['songs_str'] = min(match['songs']) + max(match['songs'])
        match_df = pd.DataFrame(matches)
        result = df.merge(match_df, on="id", how="left")
        assert(result['C_MU'].isna().sum() <= 0) # assert that the merge is correct
        if len(result) == 0:
            raise AssertionError("ids do not match! make sure they are from the same run.")
        return result
    # otherwise, return dataframe as is
    return df

def copy_file(id: str, dir:str, outname: str) -> None:
    """
    Copies files from one directory to another.
    """
    dirpath = auto_out_dir if dir == "auto" else naive_out_dir
    fp = f"{dirpath}/{id}.wav"
    shutil.copyfile(fp, f"out/survey/{outname}.wav")


def create_directory() -> None:
    """
    Main method that generates the directory needed for an ABX survey.
    """
    # first load in the two directories
    naive_df = load_df(naive_eval_path, naive_match_path)
    auto_df = load_df(auto_eval_path, auto_match_path)
    # then, we wnat to take the top and bottom of auto
    top = auto_df.head(THRESHOLD).reset_index()
    bottom = auto_df.tail(THRESHOLD).reset_index()
    # for each one, we want to find (1) its flip in auto, and (2) its corresponding mashup in naive
    ends = {"top": top, "bottom": bottom}
    out = "out/survey/output_songs.txt"
    print("Iterating through the files...")
    with open(out, 'w') as f:
        for name, end in ends.items():
            f.write(f"Compiling for {name}...\n")
            for i in tqdm(range(THRESHOLD), desc=name):
                f.write(f"#### TOP {i} ####\n")
                f.write(f"SONG:\t{end.loc[i]['id']}\n")
                copy_file(end.loc[i]['id'], "auto", f"{name}_{i}_main")
                # look for its counterpart in auto
                song_str = end.loc[i]['songs_str']
                counterpart = auto_df[(auto_df['songs_str'] == song_str) & (auto_df['id'] != end.loc[i]['id'])]
                f.write(f"COUNTERPART:\t{counterpart['id'].values[0]}\n")
                copy_file(counterpart['id'].values[0], "auto", f"{name}_{i}_counter")
                naive_match = naive_df[(naive_df['songs_str'] == song_str)]
                f.write(f"NAIVE:\t{naive_match['id'].values[0]}\n")
                copy_file(naive_match['id'].values[0], "naive", f"{name}_{i}_naive")
    print("All done.")

            
if __name__ == "__main__":
    create_directory()