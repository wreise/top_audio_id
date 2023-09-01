import cv2
import numpy as np

#from numbers import Number
from scipy.ndimage import maximum_filter, gaussian_filter

class BagOfFeatures():

    def __init__(self, factory, detector, patchesExtractor, freqPenalty = None, normalizePatches = False, normalizeFreqs=False):
        self.factory = factory
        self.detector = detector
        self.patchesExtractor = patchesExtractor
        self.freqPenalty= freqPenalty
        self.normalizePatches = normalizePatches
        self.normalizeFreqs=normalizeFreqs

    def __str__(self):
        return 'BOF ' + self.factory.__str__()

    def printShort(self):
        return 'BOF' + self.factory.printShort()

    def printLong(self):
        return 'BOF:\n' + self.factory.__str__() + self.detector.printLong() + self.patchesExtractor.printLong()

    def oneDOutput(self):
        if self.factory is None:
            return True
        else:
            return False

    def fingerprint(self, spec):
        freq, time = self.detector(spec)
        #patches = self.patchesExtractor(spec, freq, time) #returns a dictionary of patches
        indPatches = self.patchesExtractor(spec, freq, time, returnIndices=True)
        patches = {ft: spec[tuple(indPatches[ft])].transpose() for ft in indPatches}
        if self.normalizePatches:
            patches = {ft: ( patches[ft] - np.min(patches[ft]) ) / ( np.max(patches[ft]) - np.min(patches[ft]) ) for ft in patches}
        if self.factory is not None:
            return {ft : self.factory.fingerPrintFromSpectro(patches[ft]) for ft in patches}
        else:
            return patches

    def getDistance(self, fBig, fSmall, **kwargs):
        """ Find fSmall in fBig """
        if((not fBig) or (not fSmall)):
            return np.nan
        if(self.factory is not None):
            dist = [[[self.frequencyPenalty(ft1[0], ft2[0])*self.factory.getDistance(fBig[ft1][dim], fSmall[ft2][dim], **kwargs) for ft2 in fSmall] for ft1 in fBig] for dim in range(0,2)]
            return np.asarray([distanceMatching(dist[dim]) for dim in range(0,2)]) #np.mean(np.min(np.asarray(dist),axis=1),axis=1)
        else:
            def distance(x,y):
                #print(x,y)
                v = np.zeros( (max(x.shape[0], y.shape[0]), max(x.shape[1], y.shape[1])) )
                v[:x.shape[0],:x.shape[1]] = x
                v[:y.shape[0],:y.shape[1]] = v[:y.shape[0],:y.shape[1]] -y
                return np.linalg.norm(v, 'fro')
            dist = [[self.frequencyPenalty(ft1[0], ft2[0])*distance(fBig[ft1], fSmall[ft2]) for ft2 in fSmall] for ft1 in fBig]
            return distanceMatching(dist)
    
    def frequencyPenalty(self, f1,f2):
        if self.freqPenalty is None:
            return 1
        sigma = self.freqPenalty
        return np.exp(abs(f1-f2)/(2*sigma**2))

    def fingerprintToAnnoyVectorConcat(self, fingerP):
        if(len(fingerP)>0):
            frequencies = np.array([f[0] for f in fingerP])[:,None]
            tmp = [self.factory.featureToAnnoyVector(fingerP[f]) for f in fingerP]
            return np.concatenate([frequencies, tmp], axis = 1)
        else:
            return []

    def fingerprintToAnnoyVectorConcatEqualWeight(self, fingerP):
        if(len(fingerP)>0):
            frequencies = np.array([f[0]/132 for f in fingerP])[:,None]
            tmp = [self.factory.featureToAnnoyVector(fingerP[f]) for f in fingerP]
            return np.concatenate([frequencies, tmp], axis = 1)
        else:
            return []

    def fingerprintToAnnoyVectorPH(self, fingerP, dim):
        if(len(fingerP)>0):
            return np.array([self.factory.featureToAnnoyVector(fingerP[p][dim]) for p in fingerP])
        else:
            return []

    def fingerprintToAnnoyVectorFreq(self, fingerP):
        if(len(fingerP)>0):
            return [[p[0]] for p in fingerP]
        else:
            return []

from scipy.optimize import linear_sum_assignment as ls
def distanceMatching(dist):
    #return np.mean(np.min(np.asarray(dist), axis = 0), axis = 0)
    dist = np.asarray(dist)
    if (len(dist)==0):
        return np.nan
    row_ind, col_ind = ls(dist)
    return dist[row_ind, col_ind].sum()

class PatchesExtractor():

    def __init__(self, windowSize = {'f' : 10, 't' : 40}):
        """  windowSize = {'f': windowSizeInFrequencyDomain, 't': windowSizeInTimeDomain}
        Be aware that the effective window size can vary:
        1) if the window would not fit in the spectrogram, then only available pixels are returned,
        2) if the window fits, center - w//2, center + w//2 + 1 is returned
        """
        self.windowSize = windowSize

    def printLong(self):
        return 'Patches Extractor: \nwindowSize: {0}'.format(self.windowSize)

    def __call__(self, spec, freq, time, returnIndices = False):

        def getLimits(s, l, w):
            """ s - the length of the vector
            l - the location of the center.
            w - width of thedesired window.
            If not possible to get them, return smaller ones """
            return np.arange(max(0, l-(w//2)+1), min(s,l + (w//2)+1))

        def getLimits2D(s,f,t,w):
            return [getLimits(s[0], f, w['f']), getLimits(s[1],t,w['t'])]

        indices = {(f,t): np.meshgrid(*getLimits2D(spec.shape,f,t, self.windowSize)) for f,t in zip(freq, time)}
        #print(indices[(freq[0], time[0])])
        if returnIndices:
            return indices
        else:
            return {ind: spec[tuple(indices[ind])].transpose() for ind in indices}



def to_bytes(values):
    """
    Convert an array of normalized floats in range 0..1 to bytes, i.e.
    integers in range 0..255.

    Parameters
    ----------
    values : array_like
    values to convert to bytes. Must be a normalized float between 0 and 1.

    Returns
    -------
    bytes : np.ndarray
    byte array of the same shape as `values`
    """
    assert np.issubdtype(values.dtype, np.floating), "values must be of type float"
    assert np.min(values) >= 0, "values must be larger than or equal to zero"
    assert np.max(values) <= 1, "values must be smaller than or equal to one"
    values = values * 0xff
    # Cast and return
    return np.asarray(values, dtype=np.uint8)


class FastDetector():
    """
    FAST feature detector, [1]_ [2]_.

    Parameters
    ----------
    threshold : float
    threshold on difference between intensity of the central pixel
    and pixels of a circle around this pixel
    nonmax_suppression : bool
    if true, non-maximum suppression is applied to detected corners
    (keypoints)
    neighborhood_type : int
    one of the three neighborhoods `cv2.FAST_FEATURE_DETECTOR_TYPE_9_16`,
    `cv2.FAST_FEATURE_DETECTOR_TYPE_7_12`, or
    `cv2.FAST_FEATURE_DETECTOR_TYPE_5_8`

    References
    ----------
    .. [1] Feature detection and description. In OpenCV 2.4.13.0 documentation.
    http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html#fast
    .. [2] E. Rosten and T. Drummond. Machine learning for high-speed corner
    detection. In ECCV Proceedings 2006, pages 430--443. Springer, 2006.
    http://doi.org/10.1007/11744023_34
    """
    def __init__(self, threshold=10, nonmax_suppression=True, neighborhood_type=None, max_count=None):
        # max_count is a maxcount for a 30s second, WITH SR = 11025 and h = 256
        if neighborhood_type is None:
            neighborhood_type = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        self.threshold = threshold
        self.nonmax_suppression = nonmax_suppression
        self.neighborhood_type = neighborhood_type
        self.detector = cv2.FastFeatureDetector_create(threshold, nonmax_suppression, neighborhood_type)
        self.max_count = max_count
        self.max_length_of_spectro = 11025*30/256

    def printLong(self):
        return 'FastDetector:\nthreshold: {0},\n max_count: {1}'.format(self.threshold, self.max_count)

    def __call__(self, spectrogram):
        """
        Locate features using the FAST feature detector.

        Parameters
        ----------
        spectrogram : Spectrogram
        spectrogram to locate features in

        Returns
        -------
        frequency : np.ndarray
        frequency coordinates in pixels
        time : np.ndarray
        time coordinates in pixels
        """
        # Convert to a byte array
        #bytes = to_bytes(spectrogram.values)
        bytes = to_bytes(normalize(spectrogram))
        spectrogram_length = spectrogram.shape[1]
        # Detect features
        features = self.detector.detect(bytes)
        # Convert to a sensible format (note that the indices are swapped in opencv)
        if features:
            time, frequency, response = np.transpose([feature.pt + (feature.response,) for feature in features])
            # Restrict to a maximum number of features
            if(self.max_count is not None):
                max_count = int(self.max_count*spectrogram_length/self.max_length_of_spectro)
            if self.max_count and len(response) > max_count:
                idx = np.argsort(-response)[:max_count]
                time = time[idx]
                frequency = frequency[idx]
            return frequency.astype(int), time.astype(int)
        else:
            return np.empty(0, dtype=np.int), np.empty(0, dtype=np.int)

def normalize(x):
    v =  (x - np.min(x))
    return v/np.max(v)
