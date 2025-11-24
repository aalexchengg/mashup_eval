#!/bin/bash
# Setup file for creating the environment in a Linux HPC.

# Generate directories to populate files.
mkdir data
mkdir out

echo "Creating Environment...."
# Load in global conda
module load anaconda3/2024.10-1
# create the conda environment
conda create -n mashup && conda activate mashup
conda install python=3.11
conda install ffmpeg
pip install -r requirements.txt

echo "Downloading Data..."

echo "Downloading FMA Small..."
echo "This will take a while..."
FMA_DIRECTORY="data/fma_small"
if [ ! -d $FMA_DIRECTORY ]; then
  wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
  unzip fma_small.zip
  mv fma_small data/fma_small
  mv fma_small.zip data
else
  echo "Directory '$FMA_DIRECTORY' exists. Skipping this part."
fi


echo "Downloading sample data...."
SAMPLE_DIRECTORY="data/sample"
if [ ! -d $SAMPLE_DIRECTORY ]; then
  gdown 1V_LDfm_GT2_XCN5EeXrzA7xBifxkxDAu
  unzip fma_subset.zip
  # cleanup
  mv music_subselection data/sample
  rm fma_subset.zip
else
  echo "Directory '$SAMPLE_DIRECTORY' exists. Skipping this part."
fi

echo "Downloading COCOLA model..."
MODEL_DIRECTORY="data/cocola_model"
if [ ! -d $MODEL_DIRECTORY ]; then
  gdown 1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ
  mkdir $MODEL_DIRECTORY
  mv checkpoint-epoch=87-val_loss=0.00.ckpt $MODEL_DIRECTORY
else
  echo "Directory '$MODEL_DIRECTORY' exists. Skipping this part."
fi

echo "Downloading preprocessed data..."
PREPROCESS_DIRECTORY="data/auto_preprocess"
if [ ! -d $PREPROCESS_DIRECTORY ]; then
  gdown 1eGWDzdWef4wMLTrRL05qwAFlMPGzbAX3
  gdown 1mzQb_zzlKLIgJrQsCBtkD-3rtod7_pzb
  unzip out-20251113T044448Z-1-001.zip -d preprocess_out_0
  unzip out-20251113T044448Z-1-002.zip -d preprocess_out_1
  mkdir $PREPROCESS_DIRECTORY
  cp -r preprocess_out_0/out/* $PREPROCESS_DIRECTORY
  cp -r preprocess_out_1/out/* $PREPROCESS_DIRECTORY
  # cleanup
  rm -rf preprocess_out_0
  rm -rf preprocess_out_1
  rm out-20251113T044448Z-1-001.zip
  rm out-20251113T044448Z-1-002.zip
else
  echo "Directory '$PREPROCESS_DIRECTORY' exists. Skipping this part."
fi

echo "Generating the holdout set..."
python -m src.prepare.prepare_holdout_dir -p 0.01 -s data/holdout_set_mini

echo "Environment Setup Complete!"