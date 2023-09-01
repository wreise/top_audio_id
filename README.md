# README
This repository contains the code for the topological audio fingerprint and identification methods from the paper "Topological fingerprints for audio identification" (Reise*, Fernandez*, Dominguez, Harrington and Beguerisse-Diaz, 2023).

## Installation and use
The package and the library are in python. The following should get you started to create an environment, install the package and all the dependencies with conda.
```
conda create -n audioId python=3.8
conda activate audioId
python -m pip install -e .
```

You can checkout the notebook in `./notebooks/`.

If librosa throws some error (`NoBackendError`), you might need to get codecs to load the songs. In that case, if you're using conda, this should fix it
```
conda install -c conda-forge ffmpeg
```
Make sure to restart the notebook kernel though.

