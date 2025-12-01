# @author abcheng. used to analyze evaluation results.
import pandas as pd
from typing import Optional, Dict, Tuple
import numpy as np
import json
from collections import defaultdict
import argparse
import logging


logger = logging.getLogger(__name__)

# TODO: make this an argparse
# evaluate filepaths
naive_eval_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/naive_eval_output.csv"
auto_eval_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/auto_eval_output.csv"

# match filepaths
naive_match_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/eval_out/psc_naive_match.jsonl"
auto_match_path = "/Users/abcheng/Documents/workspace/mashup_eval/out/cocola_out/full_match_out.jsonl"

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


def get_composite_scores(match_filepath: str) -> Dict[int, float]:
    """
    Gets a composite matching score based on COCOLA output.\\
    composite_score = (vocalAinstrumentB + vocalBinstrumentA) / 2.\\
    @param match_filepath: path to cocola scores.\\ 
    @returns dictionary of key: hash value of two songs in sorted order, value: composite score.\\
    """
    result = defaultdict(lambda: 0)
    with open(match_filepath, 'r') as f:
        matches = [json.loads(line) for line in f]    
    for match in matches:
        match['songs_str'] = min(match['songs']) + max(match['songs'])
        hash_value = hash(match['songs_str'])
        if hash_value in result:
            result[hash_value] = (result[hash_value] + match['score']) / 2
        else:
            result[hash_value] = match['score']
    return result

def get_rank_correlation(df: pd.DataFrame, colA: str, colB: str) -> Tuple[float, float]:
    """
    Gets the rank correlation between two columns. Requires columns to be numeric type.\\
    @param df: dataframe to do rank correlation on.\\
    @param colA: first column to do rank correlation on.\\
    @param colB: second column to do rank correlation on.\\
    @returns spearman correlation, kendall correlation as a tuple.
    """
    pearson_rank_correlation = df[colA].corr(df[colB], method='spearman')
    kendall_rank_correlation = df[colA].corr(df[colB], method='kendall')

    logger.info(f"Spearman Rank Correlation: {pearson_rank_correlation}")
    logger.info(f"Kendall Rank Correlation: {kendall_rank_correlation}")

    return pearson_rank_correlation, kendall_rank_correlation

def main():
    # TODO: write results to a text file.
    naive_df = load_df(naive_eval_path, naive_match_path)
    auto_df = load_df(auto_eval_path, auto_match_path)
    composite_scores = get_composite_scores(auto_match_path)
    naive_df['score'] = naive_df['songs_str'].apply(lambda x: composite_scores[hash(x)])
    # flag the source
    naive_df['source'] = '0'
    auto_df['source'] = '1'
    # stack together
    stacked_df = pd.concat([naive_df, auto_df], axis = 0)
    # NAIVE CORRELATIONS
    logger.info("naive rank correlations for score vs CMU")
    get_rank_correlation(naive_df, 'score', 'C_MU')
    logger.info("naive rank correlations for score vs NLL")
    get_rank_correlation(naive_df, 'score', 'NLL')
    # AUTO CORRELATIONS
    logger.info("auto rank correlations for score vs CMU")
    get_rank_correlation(auto_df, 'score', 'C_MU')
    logger.info("auto rank correlations for score vs NLL")
    get_rank_correlation(auto_df, 'score', 'NLL')
    # STACKED CORRELATIONS
    logger.info("combined rank correlations for score vs CMU")
    get_rank_correlation(stacked_df, 'score', 'C_MU')
    logger.info("combined rank correlations for score vs NLL")
    get_rank_correlation(stacked_df, 'score', 'NLL')
    # check ranking scores
    logger.info("correlation with all 0s and all 1s?")
    get_rank_correlation(stacked_df, 'source', 'C_MU')




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()