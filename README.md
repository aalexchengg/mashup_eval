# mashup_eval
Automatic Evaluation of Machine Generated Mashups

# Setup

## Create the conda environment

```
conda create -name mashup && conda activate mashup
conda install python=3.11
```

## Install packages

First, install the rest of the dependencies like so.
```
pip install -r requirements.txt
```
Install allin1 is actually a bit of headache but it's ok, we can get through this together. You can find the official instructions [here](https://pypi.org/project/allin1/), but for your sake as well as mine I will recreate them in this README.

You will first have to install madmom from the GitHub repository for a proper installation of allin1
```
pip install git+https://github.com/CPJKU/madmom
```
Then, install **this specific version of natten** (they love to break backwards compatibility every update)

```
First, install the rest of the dependencies like so.
```
pip install natten==0.17.3+torch220cpu -f https://shi-labs.com/natten/wheels
```

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

# Running Experiments