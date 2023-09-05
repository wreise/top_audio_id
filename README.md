# README
This repository contains the code for the topological audio fingerprint and identification methods from the paper "Topological fingerprints for audio identification" (Reise, Fernandez, Dominguez, Harrington and Beguerisse-Diaz, 2023).

## Contents
The repository contains the implementation of the proposed topological and the Shazam (benchmark) algorithm.  The latter is based on [the open-source implementation](https://github.com/itspoma/audio-fingerprint-identifying-python).

We also provide two notebooks (one for each of the methods), which show how to load audio tracks, compute the fingerprints and in general, the inner workings of the algorithms.

This repository is not an app for audio identification.

## Installation and use
The package and the library are in python. The following should get you started to create an environment, install the package and all the dependencies with conda.
```
conda create -n audioId python=3.9
conda activate audioId
python -m pip install -e .
```

You can checkout the notebook in `./notebooks/`.

If librosa throws some error (`NoBackendError`), you might need to get codecs to load the songs. In that case, if you're using conda, this should fix it
```
conda install -c conda-forge ffmpeg
```
Make sure to restart the notebook kernel though.

