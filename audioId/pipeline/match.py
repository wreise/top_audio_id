import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

def get_delta_t_matching(x, dimensions_weight = np.array([1,0])):
    """ Beware, the algo is not symmetric.
    The two dimensions from the same row may be assigned to different columns.
    Parameters:
        x - dict, with keys 'distances', 'keys_1', 'keys_2',
            where x['distances'] is a 3d array, [:,:,dim].
    Returns:
        1d array (distances.shape[0], ), which represents the misalignment in the matches windows. """
    # Get minimal pairs (for each row, find the minimal column)
    d = np.dot(np.array(x['distances']), dimensions_weight)
    ind_min0, ind_min1 = linear_sum_assignment(d)
    # Retrieve the keys
    min_keys_1, min_keys_2 = [np.array([np.mean(x[cn][ind]) for ind in indices])
                              for cn, indices in zip(['keys_1', 'keys_2'], [ind_min0, ind_min1])]
    # Calculate the misalignent in keys, on the middle value
    return min_keys_1, min_keys_2, np.array(min_keys_1-min_keys_2)


def get_relationship_score(matching, k=5):
    x, y = matching[0], matching[1]
    smoothed = median_filter(y, mode="nearest", size=(k,))
    return pearsonr(x, smoothed).statistic

def get_error_from_matching(matching, k=5):
    return 1.-get_relationship_score(matching, k=k)


# ----- Utils -----

def get_cumulative(values):
    """ Cumulative distribution of values.
    Params:
        values: 1d np array,
    Returns x,y, where:
        x: ordinate values,
        y: a discretization of [0,1]
    """
    x = np.sort(values)
    y = np.linspace(0,1, values.shape[0])
    return x,y
