import numpy as np
from functools import partial

def intensity(image):
    return image

def intensity_at_levels(image, levels):
    """ Intensity filtration at discrete levels. In particular, all the pixels with values [level_k, level_{k+1}[ will have value level_{k}."""
    assert(all(np.diff(levels)>0)) # test if it is sorted
    assert(np.min(image)>=np.min(levels))
    index = np.argmin(image[:,:,np.newaxis]>=levels, axis = 2) - 1 # the values bigger than max(levels) have value -1, and those that res
    return levels[index]

def time(image):
    return np.tile(np.arange(image.shape[1], dtype = float),
                   reps = (image.shape[0],1))

def time_at_levels(image, levels):
    return intensity_at_levels(time(image), levels)

def frequency(image, fct = time):
    return np.flipud(np.rot90(fct(np.rot90(image, k = 3)), k = 1)) #np.rot90(fct(np.rot90(image, k = 1)), k = 3)

frequency_at_levels = lambda image, levels: frequency(image, fct = partial(time_at_levels, levels = levels))

def map_bifiltration(bif, fct):
    return map(fct, bif)

def bifiltration(image, continuous, discrete, super_level, fill_value = None):
    """ Compute the bifiltration from an image.
    Params:
        image - np.ndarray, 2d
        continuous - fct (image -> filtration image)
        discrete - same type as continuous,
        super_level - boolean, indicating whether the super-level function needs to be,
        fill_value - a np.float: the value to use, if the pixel is not present (fitlered out with the discrete filtration)
    Returns:
        all_filtrations: a list of filter functions (or, a 3d array (nb_levels, image.shape)), on the image. The vertices which are not present, from the filtering, carry +/-np.inf.
    """
    if fill_value is None:
        fill_value = np.NINF if super_level else np.INF
    discrete_filtered = discrete(image)
    all_levels = np.unique(discrete_filtered) # the output is sorted
    #all_levels = all_levels[:-1] if super_level else all_levels[0:] # get rid of the trivial level, where there is no pixel
    select_fct = lambda level: (discrete_filtered>=level if super_level else image[discrete_filtered>=level])
    all_filtrations = np.stack([np.where(select_fct(level), continuous(image), fill_value)
                                     for level in all_levels])
    return all_filtrations

intensity_frequency = partial(bifiltration,
                              continuous = intensity,
                              discrete = frequency, super_level = True)

intensity_time = partial(bifiltration,
                              continuous = intensity,
                              discrete = time, super_level = True)
