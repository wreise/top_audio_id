# audio_id_tda
Code for audio processing and identification using persistent homology. 

## Installation and use
The package and the library are in python. The following should get you started to create an environment, install the package and all the dependencies with conda.
```
conda create -n audioId python=3.8
conda activate audioId
python -m pip install -e .
```

You can checkout the notebook in `./notebooks/`.

If librosa throws some error (`NoBackendError`), it might be because you do not have the appropriate codecs to load the songs. In that case, if you're using conda, this should fix it
```
conda install -c conda-forge ffmpeg
```
Make sure to restart the notebook kernel though.


## Data

`data/oxford_dataset_downloaded` is a pandas dataframe with data about the downloaded songs

`data/one_vs_one_pairs` is a pandas dataframe with a list of ((song1, song2), Y) pairs, where Y=0,1. We set Y=1 if song2 = T(song1), where T(\cdot) is a transformation (pitch shifting, adding noise, ... )


## Final thoughts

Thanks again for joining and don't hesitate to contact me/open an issue here. Also, you probably know about the *Geometry & Topology in ML* slack channel: I sometimes use it as an alternative to collaborate. [Here is the invite link](https://join.slack.com/t/geometry-topology-ml/shared_invite/zt-1p6oo785e-l5XXoJEZIOPIAXMsMjrEYQ)
