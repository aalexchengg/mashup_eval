# Matchers
This subdirectory implements song matchers, in which all classes inherit from `base_matcher.py`.
To get started with any generator, we call `create()`, similar to from_pretrained in HuggingFace.
```
from src.generators.base_matcher import BaseMatcher
matcher = BaseMatcher.create('naive')
```
A feature of our setup is that BaseMatcher can take any type of name, as long as it contains in its string one of our generator subclasses. So for example
```
matcher = BaseMatcher.create('naive_1')
matcher = BaseMatcher.create('naive_2')
```
Would create two unique matchers with unique output directories (unless otherwise specified). This is useful when you want to use the same matcher but tune different hyperparameters.

The most important function of any class that inherits from BaseMatcher is `generate_matches()`, which given an audio direcotry and argument specifications will generate a `jsonl` file of all possible Matches.
# Generating Matches

To generate possible matches, run the following command
```
python3 generate_matches.py -inp_dir [path\to\audio_samples] \
-matcher [default='naive', 'cocola'] \
-sort [default='unsorted', 'largest', 'smallest'] \
-max_size [default=-1] \
-out_dir [default='{matcher}_out'] \
-out_path [default= '{matcher}_out/match_out']
```
Of the flags, only `-inp_dir` is required; the rest all have default values. 