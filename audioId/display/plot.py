import librosa.display as dsp

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap

COLORMAP = Colormap('coolwarm')

specshow_args = {'x_coords': None, 'y_coords': None,
                'x_axis': None, 'y_axis':None, 'sr': 22050, 'hop_length': 512,
                'fmin':None, 'fmax':None, 'tuning':0.0, 'bins_per_octave':12, 'ax':None,
                'cmap': 'gnuplot2'#'nipy_spectral' #'magma'
                }
SPECTRO_SIZE = (20, 5)

def show_spectrogram(spec, **kwargs):
    """ Show spectrogram
    Params:
        spec: core.Spectrogram
    Return:
        figure (?) a list of axes (?)"""
    purged_kwargs = {c: kwargs[c] for c in kwargs if c in specshow_args}
    if 'cmap' not in purged_kwargs:
        purged_kwargs.update({'cmap': specshow_args['cmap']})
    dsp.specshow(data = spec.spec, **purged_kwargs)
    plt.colorbar(format='%+2.2f dB')#, mappable = ScalarMappable(norm = None, cmap = COLORMAP))
    plt.title('{0}-frequency spectrogram'.format(kwargs['y_axis']))
    #(data, x_coords=None, y_coords=None, x_axis=None, y_axis=None, sr=22050, hop_length=512, fmin=None, fmax=None, tuning=0.0, bins_per_octave=12, ax=None, **kwargs)
    #return 1

def show_waveplot(a):
    dsp.waveplot(y = a.values, sr = a.sr)


def one_plot(fct, arg, figsize = SPECTRO_SIZE):
    return iterate_(fct, [arg], figsize = figsize)

"""def one_plot(fct, arg, figsize = SPECTRO_SIZE):
    fig = plt.figure(figsize = figsize); ax = plt.subplot(111)
    plt.sca(ax)
    fct(arg)
    plt.tight_layout()
    return ax"""


def iterate_(fct, arg, figsize = SPECTRO_SIZE, stack_vertically = True):
    nb_ = len(arg)
    if stack_vertically:
        _, ax = plt.subplots(nb_, 1, figsize = (figsize[0], nb_*figsize[1]))
    else:
        # stack horizontally
        _, ax = plt.subplots(1, nb_, figsize = (nb_*figsize[0], figsize[1]))

    ax = [ax] if nb_==1 else ax
    for ind in range(nb_):
        local = arg[ind]
        plt.sca(ax[ind])
        fct(local)
        #ax[ind].set_aspect('equal')
    plt.tight_layout()
    return ax
