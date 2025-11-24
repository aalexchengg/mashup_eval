# Generators
This subdirectory implements mashup generators, in which all classes inherit from `base_mashup_generator.py`.
To get started with any generator, we run `create()`, similar to from_pretrained in HuggingFace.
```
from src.generators.base_mashup_generator import BaseMashupGenerator
generator = BaseMashupGenerator.create('identity')
```
A feature of our setup is that BaseMashupGenerator can take any type of name, as long as it contains in its string one of our generator subclasses. So for example
```
generator = BaseMashupGenerator.create('identity_1')
generator = BaseMashupGenerator.create('identity_2')
```
Would create two unique generators with unique output directories (unless otherwise specified). This is useful when you want to use the same generator but tune different hyperparameters.

The most important function of any class that inherits from BaseMashupGenerator is `generate()`, which given a list of audio paths and argument specifications will generate a single mashup.

## Types of Generators
We currently support two different generators
- `IdentityMashupGenerator` - returns the first audio sample given to `generate()`
- `AutoMashupGenerator` - uses the approach given by [AutoMashup](https://github.com/ax-le/automashup/tree/main/automashup).

## Generating Mashups

To generate mashups, run the following command
```
python3 generate_mashups.py \
-matches [\path\to\json\list\of\matches] \
-generator [default='identity', 'auto'] \
-out_dir [default=`{generator}_out`]
```

Of the flags, only `-matches` is required; the rest all have default values.