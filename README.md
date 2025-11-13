# mashup_eval
Automatic Evaluation of Machine Generated Mashups

# Setup

## Download the FMA Dataset

You can find information about FMA [here](https://github.com/mdeff/fma). For our purposes, we use fma_small, which contains 8000 samples of music that are each 30 seconds long in a mp3 file format. While you can download the dataset by clicking on the link in the Github repository, we also give command line instructions to download it below.

```
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```
If that doesn't work, try using 7zip to unzip the file
```
7z x fma_small.zip
```

In addition, download a subset of 21 songs that we use specifically for song mashups. You can do so by doing
```
wget https://drive.google.com/file/d/1V_LDfm_GT2_XCN5EeXrzA7xBifxkxDAu/view?usp=sharing
```

## Create the conda environment

```
conda create -name mashup && conda activate mashup
conda install python=3.11
```

## Install packages

 Install the dependencies like so.
```
pip install -r requirements.txt
```

## Preprocess the subset dataset


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

Alternatively, if none of this works out for you, you can also take it from my Google Drive with the following commands
```
wget https://drive.google.com/file/d/1eGWDzdWef4wMLTrRL05qwAFlMPGzbAX3/view?usp=sharing
wget https://drive.google.com/file/d/1mzQb_zzlKLIgJrQsCBtkD-3rtod7_pzb/view?usp=sharing
unzip out-20251113T044448Z-1-001.zip
unzip out-20251113T044448Z-1-002.zip
```
Note that there are duplicates between the two folders, so make sure you only copy over the correct subdirectories from out 2 to out. The correct move should be in the commands below
```
mv out\ 2/separated/htdemucs/Dragon\ Or\ Emperor\ -\ Part\ of\ Me\ Says out/separated/htedemucs
mv out\ 2/separated/htdemucs/Los\ Steaks\ -\ Sunday\ Girls out/separated/htedemucs
mv out\ 2/separated/htdemucs/The\ Cute\ Lepers\ -\ Young\ Hearts out/separated/htedemucs
```

# Running Experiments