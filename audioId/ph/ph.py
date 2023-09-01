import gudhi
import numpy as np
from functools import partial

def compute_ph(image, super_level):
    """ Compute ph, with the filtration given by the image.
    Params:
        image: np.ndarray, representing the filtration, f.ex. np.array([[2,0],[1,3],[4,5]]),
        super_level: boolean
    """
    shape = image.shape
    tmp_image = -image if super_level else image
    complex, _ = gudhi_lower_star_from_vertices(tmp_image)
    complex.persistence(homology_coeff_field = 2, min_persistence = 0)
    return {'complex': complex, 'super_level': super_level}

def curry_super_level(fct):
    return partial(fct, super_level = True)

def curry_sub_level(fct):
    return partial(fct, super_level = False)

compute_ph_super_level = curry_super_level(compute_ph)
compute_ph_sub_level = curry_sub_level(compute_ph)

def read_pds(pd, dimension, reduce = True):
    """
    Params:

    Returns:
        """
    bd = np.array(pd['complex'].persistence_intervals_in_dimension(dimension))
    if pd['super_level']:
        bd = -bd
    if reduce and (dimension==0):
        bd = bd[:-1]
    return bd

def gudhi_lower_star_from_vertices(image):
    """Instantiate a 2d, lower-star filtration from an image.
    Create a bigger complex, with values from `image' as 0-cubes. Propagate those values to edges (horizontal_, vertical_) and to 2_cubes.
    This works thanks to assertions:
        - the values of 0-cubes in horizontal_ and vertical_ stay the same, since we take the maximum of the repeated values.
        - the rows without 2-cubes stay the same in center_.
    Params:
        image: numpy.ndarray, dim 2,
    Returns:
        complex_: lower star filtration from image,
        center_: array, with the lower-star filtration"""
    v = np.repeat(np.repeat(image, 2, axis = 0), 2, axis = 1) # expand, mimicking the datastructure
    dimensions = [2*(s-1) +1 for s in image.shape] # number of simplices to define in each dimension
    horizontal_ = np.maximum(v[:,0:-1], v[:,1:]) # filtration values for horizontal edges (and 0-cubes)
    vertical_ = np.maximum(v[0:-1,:], v[1:,:]) # filtration values for horizontal edges ( and 1-cubes)
    center_ = np.maximum(np.maximum(horizontal_[0:-1,:], horizontal_[1:,:]),
                         np.maximum(vertical_[:,0:-1], vertical_[:,1:])) # filtration values for 2-cubes
    # rows with pair indices stay the same, f.ex. horizontal_[0] ==  np.maximum(horizontal_[0:-1,:], horizontal_[1:,:])[0]
    complex_ = gudhi.CubicalComplex(dimensions =dimensions,
                                top_dimensional_cells = np.ravel(center_.transpose())) # transpose, due to the order of simplices
    return complex_, center_

