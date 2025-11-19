# Evaluation Suite

## Creating the Holdout Directory

A helper file to generate a holdout set for evaluation purposes (to be used in the future, e.g. during `holdout_set_generator`). Takes a subset of files from an unprocessed FMA dataset folder, which consists of only mp3 files. Run the following command in the root folder:

```
python -m src.evaluate.prepare_holdout_dir
```

Options: 
- `-f` `--fma-path`: path of the FMA folder (defaults to `data/fma_small`)
- `-p` `--percent`: percentage of FMA folder to take (defaults to `0.1`)
- `-s` `--save-path`: folder to save holdout set (defaults to `data/holdout_set`)

## Running the Evaluator

Our proposed evaluation metric insofar is calculated by $$C_{MU}(x) = \frac{k \cdot d_{HO}(x)}{(1-k) \cdot NLL(x)}$$

The main file, after creating a holdout set directory from FMA, can be ran as follows: 

```
python -m src.evaluate.evaluate_mashups
```

This will generate two files (a .csv and a .txt), where one is purely the data and the other is a more human-readable summary of the evaluation process. Because there is a default evaluation folder created, both of these files will have the timestamp at the end of them for easy identification of which run they were generated from.

Options: 
- `-a` `--audio-path`: path for audio files to evaluate. this is a REQUIRED field and must be either a wav file or a directory of wav files. 
- `-c` `--centroid`: boolean for whether our distance metric ($d_{HO}$) should be calculated using the distance to the nearest neighbor or the centroid of the holdout set. defaults to `True` (using centroid)
- `-o` `--output-path`: directory for where the output should be stored. defaults to `data/eval_output`. 
- `-k` `--k`: $k$ parameter for our metric, which defaults to `0.5` (has no effect on metric)
- `-p` `--path-to-holdout`: path (from root) of directory holding only and all holdout set `.wav` files. defaults to `data/holdout_set`. (note that poor choice in flag is due to the fact that `-h` is always reserved for `--help`.)

## Testing Individual Functionalities 

Most of this should be self explanatory when looking at the main functions of `holdout_set.py` and `nll_extractor.py`, but if more explanations are needed please ping me. 