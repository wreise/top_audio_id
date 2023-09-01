from audioId.core.spectrogram import MelSpectrogram, STFT
from audioId.ph.vectorization import BettiCurveExact
from audioId.ph.ph import read_pds, compute_ph_super_level
from . import match
from . import detector

from functools import partial
from itertools import product
import numpy as np
from collections import OrderedDict
from tqdm import tqdm as tqdm

WINDOWS_OVERLAP = 0.4
WINDOW_LENGTH = 1
WINDOW_HEIGHT = 30
WINDOWS_HEIGHT_OVERLAP = 0.2

DEFAULT_VECT = {0: BettiCurveExact(is_super_level = True),
                1: BettiCurveExact(is_super_level = True)}

def fingerprint_audio(audio, **kwargs):
    """ Fingerprint audio, using the filtration and vectorization method.
    Params:
        audio:
        spectrogramConstructor:
        filtration:
        compute_ph:
        vect:
    Returns:
        vectorised ph information for each window.
        The information may be a group (f.ex. when filter_fct = intensity), or a collection of groups, when using bifiltrations
    """
    windows = get_windows_from_audio(audio, kwargs['spectrogramFct'])
    return map_windows(windows, partial(window_to_ph_vector, **kwargs))

def fingerprint_audio_subimages(audio, **kwargs):
    """ Fingerprint audio, using the filtration and vectorization method.
    Params:
        audio:
        spectrogramConstructor:
        filtration:
        compute_ph:
        vect:
    Returns:
        vectorised ph information for each subimage.
        The information may be a group (f.ex. when filter_fct = intensity), or a collection of groups, when using bifiltrations
    """
    subimages = get_subimages_from_audio(audio, kwargs['spectrogramFct'])
    return map_windows(subimages, partial(window_to_ph_vector, **kwargs))

def compare_fingerprints(fa,fb, vect):
    return map2_windows(fa, fb, partial(compare_fingerprints_one, vect = vect))

def compare_audios(audio1, audio2, **kwargs):
    fa, fb = [fingerprint_audio(a, **kwargs) for a in [audio1, audio2]]
    return compare_fingerprints(fa, fb, kwargs['vect'])

def compare_audios_subimages(audio1, audio2, **kwargs):
    fa, fb = [fingerprint_audio_subimages(a, **kwargs) for a in [audio1, audio2]]
    return map2_by_freq(fa, fb, partial(compare_fingerprints_one, vect = kwargs['vect']))

def get_windows_from_audio(audio, spectrogramConstructor):
    spectro = spectrogramConstructor.from_audio(audio)
    total_time = spectro.pixel_to_time(spectro.spec.shape[1])
    starts_at = np.arange(0, total_time, WINDOW_LENGTH*(1-WINDOWS_OVERLAP))
    ends_at = np.array([s+WINDOW_LENGTH for s in starts_at])
    windows = OrderedDict([((s,en), spectro.extract_seconds(s, en))
                         for s,en in zip(starts_at,ends_at)])
    normalized_windows = map_windows(windows, lambda x: x.amplitude_to_db().normalize())
    return normalized_windows

def get_subimages_from_audio(audio, spectrogramConstructor):
    spectro = spectrogramConstructor.from_audio(audio)
    total_time = spectro.pixel_to_time(spectro.spec.shape[1])
    starts_at = np.arange(0, total_time, WINDOW_LENGTH*(1-WINDOWS_OVERLAP))
    ends_at = np.array([s+WINDOW_LENGTH for s in starts_at])
    windows = OrderedDict([((s,en), spectro.extract_seconds(s, en))
                         for s,en in zip(starts_at,ends_at)])
    #freq_pixels = np.linspace(0, spectro.spec.shape[0], NB_SEGMENTS, dtype = 'int')
    freq_pixel_start = np.arange(0, spectro.spec.shape[0], WINDOW_HEIGHT*(1-WINDOWS_HEIGHT_OVERLAP), dtype='int')
    subimages = OrderedDict([((*key, fc1), spec.extract_pixels(fc1,fc1 + WINDOW_HEIGHT)) for key, spec in windows.items() for fc1 in freq_pixel_start])
    normalized_subimages = map_windows(subimages, lambda x: x.amplitude_to_db().normalize())
    return normalized_subimages

def map_windows(windows, fct):
    return OrderedDict([ (k, fct(w)) for k,w in windows.items()])

def map2_windows(windowsA, windowsB, fct):
    return OrderedDict([((k1,k2), fct(w1,w2))
            for (k1,w1), (k2,w2) in product(windowsA.items(), windowsB.items())])

def map2_by_freq(subimagesA, subimagesB, fct):
    categories = list(set([key[2] for key in subimagesA]))
    cat_distances = {c: None for c in categories}
    for cat in categories:
        fct_filter = lambda x: {(key[0], key[1]): s for key, s in x.items() if key[2] == cat }#x[3] == cat
        cat_distances.update({cat: map2_windows(fct_filter(subimagesA), fct_filter(subimagesB), fct)})
    return cat_distances

def map_bifiltration(bifiltr, fct):
    """ Apply the fct 'fct' to every filtration in the bifiltration."""
    return map(bifiltr, fct)

def windows_to_ph(windows, filter_fct, compute_ph):
    fct = lambda x: compute_ph(filter_fct(x.spec))
    return map_windows(windows, fct)

def ph_to_vector(pds_windows, vect = DEFAULT_VECT):
    return {d: vect[d].vectorize(read_pds(pd = pds_windows, dimension = d)) for d in range(2)}

def window_to_ph_vector(window, filter_fct, compute_ph, vect, **kwargs):
    ph = compute_ph(filter_fct(window.spec))
    return {d: vect[d].vectorize(read_pds(pd = ph, dimension = d))
            for d in range(2)}

def lazy_filtered_window_to_ph_vector(filtered,
                                      compute_ph = compute_ph_super_level,
                                      vect= DEFAULT_VECT, **kwargs):
    return ph_to_vector(compute_ph(filtered), vect = vect)

def compare_fingerprints_one(fa, fb, vect):
    return {d: vect[d].distance(fa[d], fb[d]) for d in range(2)}


def get_matrix_from_dict(distances, fp1, fp2):
    """ Return distances as an array, fp1, fp2 are dictionaries with windows
    distances is an OrderedDict, with keys
    ((wind_start_song_1, window_end_song_1), (wind_start_song_2, window_end_song_2))
    and values that are {k: distance between persistence functionals for this pair of windows}.
    """
    m_distances = [[ [distances[(k,k2)][d] for d in range(2)] for k2 in fp2] for k in fp1]
    keys_1, keys_2 = [list(f.keys()) for f in [fp1, fp2]]
    return m_distances, keys_1, keys_2
