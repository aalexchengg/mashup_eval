#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 9:59:00
#SBATCH --job-name=main
#SBATCH --gpus=h100-80:1
#SBATCH --output=/ocean/projects/cis250057p/acheng8/mashup_eval/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis250057p/acheng8/mashup_eval/logs/%x-%j.err
#SBATCH --mail-user=acheng8


# Script to run main experiment.

if [ $# -lt 1 ]; then
  echo "Error: Need to provide experiment argument."
  echo "Usage: $0 [match/mash/evaluate] [OPTIONAL:verbose]"
  exit 1 # Exit with a non-zero status to indicate an error
fi

# PSC specific setup
# export HF_HOME=/ocean/projects/cis250057p/acheng8/.cache/huggingface # need to modify
# cd /ocean/projects/cis250057p/acheng8/mashup_eval # move to right directory


echo "Loading in environment..."
# module load anaconda3/2024.10-1 # for PSC only
conda init
conda activate mashup
echo "Environment loaded."

MATCH_INP_DIR="data/sample" # required for matching
MATCH_OUT_PATH="out/cocola_out_test/match_out.jsonl" # required for mashing
MASHUP_OUT_PATH="out/identity_out_test"


EXPERIMENT="$1"
VERBOSE="$2"
#### MATCHING ####
if [ "$EXPERIMENT" == "match" ] || [ "$EXPERIMENT" == "all" ]; then
    echo "##################NEW PHASE#####################"
    echo "##################Matching...##################"

    CMD="python3 -m src.generate_matches -inp_dir $MATCH_INP_DIR -config configs/cocola_matcher_config.yaml"
    if [ "$VERBOSE" == "verbose" ]; then
        CMD="$CMD -verbose"
    fi

    echo "Running: $CMD"
    $CMD
else
    echo "Skipping matching..."
fi

#### MASHING #####
if [ "$EXPERIMENT" == "mash" ] || [ "$EXPERIMENT" == "all" ]; then
    echo "##################NEW PHASE#####################"
    echo "##################Mashing...##################"

    CMD="python3 -m src.generate_mashups -matches $MATCH_OUT_PATH -config configs/mashup_config.yaml"
    if [ "$VERBOSE" == "verbose" ]; then
        CMD="$CMD -verbose"
    fi

    echo "Running: $CMD"
    $CMD
else
    echo "Skipping mashing..."
fi
#### EVALUATING #####
if [ "$EXPERIMENT" == "evaluate" ] || [ "$EXPERIMENT" == "all" ]; then
    echo "##################NEW PHASE#####################"
    echo "##################Evaulating...##################"

    CMD="python3 -m src.evaluate_mashups -a $MASHUP_OUT_PATH -config configs/evaluate_config.yaml"
    echo "Running: $CMD"
    $CMD
else
    echo "Skipping evaluation..."
fi

echo "Experiment Complete!"