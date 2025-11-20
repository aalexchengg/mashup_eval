# mashup_eval
Automatic Evaluation of Machine Generated Mashups

# Setup

## Preliminaries

Clone the repository and create the data and output directories.
```
git clone https://github.com/aalexchengg/mashup_eval.git
cd git clone
mkdir data
mkdir out
```

**IMPORTANT**: For all documentation, assume you are running from the root folder `mashup_eval/`.

## Quick Setup

Running the command below should do everything we describe (**except the actual preprocessing**). However, if you care to be pedantic, feel free to go through each section.
```
bash initial_setup.sh
```

## Environment Creation
### Create the conda environment and install packages
```
conda create -n mashup && conda activate mashup
conda install python=3.11
conda install ffmpeg<8
pip install -r requirements.txt
```

### Download the FMA Dataset

You can find information about FMA [here](https://github.com/mdeff/fma). For our purposes, we use fma_small, which contains 8000 samples of music that are each 30 seconds long in a mp3 file format. While you can download the dataset by clicking on the link in the Github repository, we also give command line instructions to download it below.

```
FMA_DIRECTORY="data/fma_small"
if [ ! -d $FMA_DIRECTORY ]; then
  wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
  unzip fma_small.zip
  mv fma_small data/fma_small
  # cleanup
  rm fma_small.zip
else
  echo "Directory '$FMA_DIRECTORY' exists. Skipping this part."
fi
```

In addition, download a subset of 21 songs that we use specifically for song mashups. 

```
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
```

### Preprocess the subset dataset


This is a bit of a headache, but an important step. You will have to run this on either a Linux machine or a Google Colab. If you are running this on a Linux machine, please use a separate virtual environment to preprocess the subset dataset.

Install the following packages; be very careful that you are using the version that we describe
```
conda create -name preprocess && conda activate preprocess # also for linux machines
conda install python=3.12 # also for linux machines
pip install numpy==1.26.4 torch==2.2.2 torchaudio==2.2.2
pip install natten==0.17.3+torch220cu121 -f https://shi-labs.com/natten/wheels
pip install git+https://github.com/CPJKU/madmom
pip install allin1
```

From there, edit the filepaths in `preprocess.py`, and then you can run 
```
python3 preprocess.py
```
To preprocess all the songs.

Alternatively, if none of this works out for you, you can also take it from my (Google Drive Folder)[https://drive.google.com/drive/folders/13ZmQ_NHpNQptrDNra0OR1GNkIQyzU9MK?usp=drive_link], and then unzipping them like so.
```
PREPROCESS_DIRECTORY="data/auto_preprocess"
if [ ! -d $PREPROCESS_DIRECTORY ]; then
  gdown 1eGWDzdWef4wMLTrRL05qwAFlMPGzbAX3
  gdown 1mzQb_zzlKLIgJrQsCBtkD-3rtod7_pzb
  unzip out-20251113T044448Z-1-001.zip -d preprocess_out_0
  unzip out-20251113T044448Z-1-002.zip -d preprocess_out_1
  mkdir $PREPROCESS_DIRECTORY
  cp -r preprocess_out_0 $PREPROCESS_DIRECTORY
  cp -r preprocess_out_1 $PREPROCESS_DIRECTORY
  # cleanup
  rm -rf preprocess_out_0
  rm -rf preprocess_out_1
  rm out-20251113T044448Z-1-001.zip
  rm out-20251113T044448Z-1-002.zip
else
  echo "Directory '$PREPROCESS_DIRECTORY' exists. Skipping this part."
fi
```

### Downloading the COCOLA model

You can download the COCOLA model by downloading from the following [link](https://drive.google.com/file/d/1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ/view?usp=share_link). 

We create a `cocola_model` subdirectory and place the model in `data/cocola_model/` like so

```
echo "Downloading COCOLA model..."
MODEL_DIRECTORY="data/cocola_model"
if [ ! -d $MODEL_DIRECTORY ]; then
  gdown 1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ
  mkdir $MODEL_DIRECTORY
  mv checkpoint-epoch=87-val_loss=0.00.ckpt $MODEL_DIRECTORY
else
  echo "Directory '$MODEL_DIRECTORY' exists. Skipping this part."
fi
```

## Result

At the end, you should have a directory that looks like this:

```
mashup_eval/
├─ data/
│  ├─ auto_preprocess/
│  ├─ sample/
│  ├─ fma_small/
│  ├─ cocola_model/
├─ out/
```

# Running Experiments

Our three main scripts are 
- `generate_matches.py` Iterates through an audio file directory and generates possible matches for mashups.
- `generate_mashups.py` Iterates through the matches created in the previous step and generates mashups.
- `evaluate_mashups.py` Iterates through the generated mashups and creates an evaluation score for each of them.

More information about each script can be found in the README of their associated subdirectory. In this section, we will provide information about how to run end-to-end experiments.


