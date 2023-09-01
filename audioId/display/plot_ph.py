import matplotlib.pyplot as plt
import numpy as np
from audioId.pipeline import map_windows
from audioId.ph.vectorization import BettiCurveExact

DEFAULT_BETTI = {0: BettiCurveExact(is_super_level=True), 1: BettiCurveExact(is_super_level=True)}


def plot_bcs_2d(bcs, dimension, are_same_coords=False, betti_vect=DEFAULT_BETTI, **kwargs):
    if not are_same_coords:
        bcs_same_coords = {
            d: list(betti_vect[d].same_coordinates([l[d] for l in bcs])) for d in range(2)
        }
    else:
        bcs_same_coords = bcs
    ys = np.stack([xy[1] for xy in bcs_same_coords[dimension]])
    x_coord = bcs_same_coords[dimension][0][0]
    if "y_coord" in kwargs:
        y_coord = kwargs.pop("y_coord")
    else:
        y_coord = np.arange(ys.shape[0])
    f = plt.contourf(x_coord, y_coord, ys, 10, **kwargs)
    return f
