from scipy.interpolate import interp1d
import numpy as np
from functools import partial

class BettiCurveExact():

    def __init__(self, is_super_level):
        self.is_super_level = is_super_level
        #self.vector = self.vectorize(pd)

    def vectorize(self, pd):
        """ persistence diagram -> betti curve
        Params:
            pd: np.ndarray, ndim = 2, with float births/deaths.
        Returns:
            x: np.array, 1d, coordinates of the betti curve,
            y: np.array, values for the betti curve.
        """
        s = pd.shape
        if (len(s) <2):
            return (np.array([0.,1.]), np.array([0., 0.]))
        #try:
        births, deaths = pd[:,0], pd[:,1]
        #except IndexError:
        #    return (np.array([0.,1.]), np.array([0., 0.]))
        assert(np.all(births< deaths)*(not(self.is_super_level))
                or (np.all(births>deaths)*(self.is_super_level)))
        births_v, deaths_v = np.ones(births.shape), -(1)*np.ones(deaths.shape)
        all_coordinates, all_values = np.concatenate([births, deaths]), np.concatenate([births_v, deaths_v])
        order = np.argsort(all_coordinates)
        n_order = np.flip(order) if self.is_super_level else order

        x, v = [ a[n_order] for a in [all_coordinates, all_values]]
        y = np.cumsum(v)
        return x,y

    def same_coordinates(self, xyList):
        """ Represent a list of Betti curves, on the same coordinates.
        Params:
            xyList: an iterable of Betti curves xy = (x,y), where x is a 1d array
        Returns:

        """
        all_coordinates = np.concatenate([np.ravel(xy[0]) for xy in xyList])
        sorted_coordinates = np.sort(np.unique(all_coordinates))
        return self.over_coordinates(xyList, sorted_coordinates)

    def over_coordinates(self, xyList, sorted_coordinates):
        """ Represent a list of Betti curves, on the same coordinates.
        We require sorted_coordinates.
        Params:
            xyList: an iterable of Betti curves xy = (x,y), where x is a 1d array,
            sorted_coordinates: an array of sorted, unique coordinates
        Return:
            iterable over new betti curves
        """
        assert((sorted_coordinates[1:] >= sorted_coordinates[:-1]).all())
        #zeros_fill = np.zeros(sorted_coordinates.shape)

        def interp_fct(x_old, y_old):
            """ From a piecewise constant function, defined by x_old, y_old,
            interpolate to sorted_coordinates.
            """
            try:
                interpolated = interp1d(x_old, y_old,
                        fill_value = 0,
                        kind = 'next' if self.is_super_level else 'previous',
                        bounds_error = False)(sorted_coordinates)
            except ValueError:
                # happens when there are no pts in the pd.
                interpolated = np.zeros(sorted_coordinates.shape)
            return (sorted_coordinates, interpolated)
        for xy in xyList:
            yield interp_fct(xy[0], xy[1])

    def distance(self, pd1, pd2):
        """ Calculate the distance between two Betti curves."""
        pd1_same, pd2_same = self.same_coordinates([pd1, pd2])
        return self._distance_over_coordinates(pd1_same, pd2_same, pd1_same[0])

    def _distance_over_coordinates(self, bn1, bn2, coords):
        p1, p2 = [p[1][1:] if self.is_super_level else p[1][0:-1]
                  for p in [bn1, bn2]]
        difference = np.abs(p1 - p2)
        return np.sum(np.diff(coords)* difference)

class BettiCurveApproximation(BettiCurveExact):

    def __init__(self, is_super_level, coordinates):
        super(BettiCurveExact, self).__init__( is_super_level)
        self.xcoords = coordinates

    def vectorize(self, pd):
        bc = super().vectorize(pd)
        return self.exact_to_approximate(bc)

    def exact_to_approximate(self, bc):
        v = self.over_coordinates([bc], self.xcoords)
        return v[0]

    def distance(self, pd1, pd2):
        return self._distance_over_coordinates(pd1, pd2, self.xcoords)
