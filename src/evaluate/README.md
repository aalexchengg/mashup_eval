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

## Testing the Embedding Generator

The embedding generator currently references MERT to convert an audio sample to the embedding space. As MERT has 13 layers, we assume the very last one represents the embedding space we want to be in. You can see this in isolation by running the following command in the root folder: 

```
python -m src.evaluate.embedding_generator
```

## Using a Holdout Set to Get Distances

The `HoldoutSet` class currently provides functionality to get the distance of some audio file (both in- and out-of-distribution of the holdout set) that we can use for a metric: either getting the Euclidean distance from the centroid of the holdout set, or the distance to the nearest neighbor within the holdout set. Test this functionality by running the following command in the root folder: 

```
python -m src.evaluate.holdout_set_generator -l data/[HOLDOUT_SET_DIRECTORY] -a data/[PATH/TO/WAV/AUDIO/FILE]
```

You can sanity check that the distance to nearest neighbor of a file within the holdout set is zero (as we are testing it against itself). You can also sanity check that the caching is in fact happening by ensuring that `hits` goes up after each cache info is stored. (Note that this only applies to same file paths, not necessarily same files that are named the same but in different directories.)

Options: 
- `-l` `--list-path`: path of directory that holds entire holdout set (nothing else) (defaults to `data/holdout_set`)
- `-a` `--audio-path`: path of the audio file being tested 

