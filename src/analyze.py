# @author abcheng. used to analyze evaluation results.
import pandas as pd
from typing import Optional, Dict, Tuple, List
import numpy as np
import os
import json
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import yaml
import logging
import io


logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-action', type = str,
                        choices = ['all', 'plot', 'corr'],
                        default = 'all',
                        help = "Analysis actions to do.")
    parser.add_argument('-config', type = str,
                        required = True,
                        help = "accepts yaml config files as well.")
    return parser

def _extract_id(df: pd.DataFrame) -> pd.DataFrame:
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
    df = _extract_id(df)
    # should also load in the match scores, if available
    if match_filepath:
        with open(match_filepath, 'r') as f:
            matches = [json.loads(line) for line in f]
        # need to coerce into a string first to remove nans
        for match in matches:
            match['songs_str'] = min(match['songs']) + max(match['songs'])
        match_df = pd.DataFrame(matches)
        result = df.merge(match_df, on="id")
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

def _get_rank_correlation(df: pd.DataFrame, colA: str, colB: str) -> Tuple[float, float]:
    """
    Gets the rank correlation between two columns. Requires columns to be numeric type.\\
    @param df: dataframe to do rank correlation on.\\
    @param colA: first column to do rank correlation on.\\
    @param colB: second column to do rank correlation on.\\
    @returns spearman correlation, kendall correlation as a tuple.
    """
    spearman_rank_correlation = df[colA].corr(df[colB], method='spearman')
    kendall_rank_correlation = df[colA].corr(df[colB], method='kendall')

    logger.info(f"Spearman Rank Correlation: {spearman_rank_correlation}")
    logger.info(f"Kendall Rank Correlation: {kendall_rank_correlation}")

    return spearman_rank_correlation, kendall_rank_correlation


def _plot_scores_single(df: pd.DataFrame, col: str, out: str) -> None:
    """
    Plots a histogram for a single column from a single dataframe source.\\
    @param df: dataframe to plot.\\
    @param col: column that we are plotting.\\
    @param out: output path, to be placed in out/analyze/.
    """
    if col not in df.columns:
        raise AssertionError(f"column {col} was not found in the columns!")
    plt.hist(df[col], bins='auto', color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Column {col} for {out}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Save the figure to a file
    plt.savefig(f'out/analyze/{out}_{col}.png', dpi=300, bbox_inches='tight')  # dpi=300 for high quality
    plt.close()  # Close the figure to free memory

def _plot_scores_overlay(df1: pd.DataFrame, label1: str, df2: pd.DataFrame, label2: str, col: str, out: str) -> None:
    """
    Plots overlayed histograms for a single column from two dataframe sources.\\
    @param df1: first dataframe to plot.\\
    @param label1: label for df1.\\
    @param df2: second dataframe to plot.\\
    @param label2: label for df2.\\
    @param col: column that we are plotting.\\
    @param out: output path, to be placed in out/analyze. 
    """
    if col not in df1.columns or col not in df2.columns:
        raise AssertionError(f"column {col} was not found in one of the columns!")
    plt.hist(df1[col], bins='auto', label = label1, color='bisque', edgecolor='black', alpha = 0.5)
    plt.hist(df2[col], bins='auto', label = label2, color='lightseagreen', edgecolor='black', alpha = 0.5)
    plt.title(f'Histogram of Column {col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Save the figure to a file
    plt.savefig(f'out/analyze/{out}_{col}.png', dpi=300, bbox_inches='tight')  # dpi=300 for high quality
    plt.close()  # Close the figure to free memory


def get_correlations(df: pd.DataFrame, col: str, name: str) -> Tuple[Tuple[float, float]]:
    """
    Generates the correlations for a certain dataframe.\\
    @param df: the dataframe to generate the correlation on.\\
    @param name: the name of the dataframe.\\
    @returns (cmu_spearman_corr, cmu_kendall_corr), (nll_spearman_corr, nll_kendall_corr).
    """
    logger.info(f"{name} rank correlations for score vs CMU")
    cmu_corr = _get_rank_correlation(df, col, 'C_MU')
    logger.info(f"{name} rank correlations for score vs NLL")
    nll_corr = _get_rank_correlation(df, col, 'NLL')
    return cmu_corr, nll_corr

def write_correlations(correlations: Tuple[Tuple[float]], name: str, out: io.TextIOWrapper) -> None:
    """
    Writes correlations to output file.
    @param correlations: output of get_correlations
    @param name: name of the dataframe
    @param out: iowrapper to write output to.
    """
    cmu_spearman, cmu_kendall = correlations[0]
    nll_spearman, nll_kendall = correlations[1]
    out.write(f"{name} cmu spearman: {cmu_spearman}\n")
    out.write(f"{name} cmu kendall: {cmu_kendall}\n")
    out.write(f"{name} nll spearman: {nll_spearman}\n")
    out.write(f"{name} nll kendall: {nll_kendall}\n")

def analyze_correlations(dfs: Dict[str, pd.DataFrame]) -> None:
    with open("out/analyze/correlations.txt", "w") as f:
        for name, df in dfs.items():
            correlations = get_correlations(df, 'score', name)
            write_correlations(correlations, name, f)


def plot_scores(dfs: Dict[str, pd.DataFrame], col: str, overlays: Optional[List[Tuple]] = None) -> None:
    """
    Plots the score distributions for naive, auto, and stacked, and also overlays auto and stacked.
    @param dfs: the dataframe dictionary that we are analyzing.
    @param overlays: pairs of overlays to generate histograms for
    """
    for name, df in dfs.items():
        _plot_scores_single(df, col, name)
    if overlays:
        for name1, name2 in overlays:
            if name1 not in dfs:
                raise AssertionError(f"{name1} is not present in the dataframe dictionary!")
            if name2 not in dfs:
                raise AssertionError(f"{name2} is not in the dataframe dictionary!")
            _plot_scores_overlay(dfs[name1], name1, dfs[name2], name2, col, f'overlay_{name1}_{name2}')


def check_attr(args) -> None:
    """
    Checks that arguments were parsed correctly.
    """
    assert(hasattr(args, "naive_eval_path"))
    assert(hasattr(args, "auto_eval_path"))
    assert(hasattr(args, "naive_match_path"))
    assert(hasattr(args, "auto_match_path"))
    assert(hasattr(args, "plot_col"))


def main(args):
    naive_df = load_df(args.naive_eval_path, args.naive_match_path)
    auto_df = load_df(args.auto_eval_path, args.auto_match_path)
    composite_scores = get_composite_scores(args.auto_match_path)
    naive_df['score'] = naive_df['songs_str'].apply(lambda x: composite_scores[hash(x)])
    # flag the source
    naive_df['source'] = '0'
    auto_df['source'] = '1'
    # stack together
    stacked_df = pd.concat([naive_df, auto_df], axis = 0)
    dfs = {"naive": naive_df, "auto": auto_df, "stacked": stacked_df}
    overlays = [('naive', 'auto')]
    # do something!
    if args.action == "corr" or args.action == "all":
        analyze_correlations(dfs)
    if args.action == "plot" or args.action == "all":
        plot_scores(dfs, args.plot_col, overlays)


if __name__ == "__main__":
    # create output directory if it doesnt exist.
    os.makedirs("out/analyze", exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = setup_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # Overwrite args with config values
    for key, value in config.items():
        setattr(args, key, value)
    check_attr(args)
    main(args)
