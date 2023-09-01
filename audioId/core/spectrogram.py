import librosa as lb
import numpy as np
import warnings
from functools import partial

from ..display import plot as dpl

NB_BINS = 128  # change to 256 as in Spotify ?
BINS_PER_OCTAVE = 6
WINDOW_SIZE = 1024  # 512 # 23ms at sr = 22050
HOP_LENGTH = 256  # recommended WINDOW_SIZE//4
F_MIN, F_MAX = 1, None

# NB_BINS = np.ceil(np.log2(F_MAX/F_MIN)*BINS_PER_OCTAVE)


class Spectrogram:
    @classmethod
    def from_audio(self, audio):
        raise NotImplementedError("Calling Spectrogram.from_audio, which is not implemented")

    def pixel_to_time(self, p):
        pass

    def time_to_pixel(self, t):
        pass

    def pixel_to_frequency(self, p):
        pass

    def frequency_to_pixel(self, f):
        pass

    def plot(self):
        pass

    def map(self, f):
        d = self.__dict__
        all_ = {c: cv for c, cv in d.items() if c != "spec"}
        return type(self)(spec=f(self.spec), **all_)

    def extract_seconds(self, t_start, t_end):
        """Extract the interval tStart-tEnd from the spectro
        Params:
            t: float, in seconds
        Return:
            instance of the correct class, but with an offset.
        """
        time0, time1 = [self.time_to_pixel(t) for t in [t_start, t_end]]
        f = lambda x: x[:, time0:time1]
        local_spectro = self.map(f)
        local_spectro.offset = t_start
        return local_spectro

    def extract_frequency(self, f_start, f_end):
        """ """
        f0, f1 = [self.frequency_to_pixel(f) for f in [f_start, f_end]]
        return self.extract_pixels(f0, f1)

    def extract_pixels(self, p_start, p_end):
        """Extract a vertical segment.
        Params:
            p_start, p_end: integers, representing pixels.
        Returns:
            local_spectro: extracted subimage."""
        local_spectro = self.map(lambda x: x[p_start:p_end, :])
        local_spectro.offset_freq = self.pixel_to_frequency(p_start)
        return local_spectro

    def amplitude_to_db(self, fct=np.max):
        return self.map(lambda x: lb.amplitude_to_db(np.abs(x), ref=fct))

    def normalize(self):
        def normalizer(x):
            m = np.max(x) - np.min(x)
            if m == 0:
                return x * 0
            else:
                return (x - np.min(x)) / m

        return self.map(normalizer)


class STFT(Spectrogram):
    def __init__(self, spec, sr, hop_length, n_fft, offset=0, **kwargs):
        self.spec = spec
        self.sr, self.hop_length, self.n_fft = (
            sr, hop_length, n_fft,
        )
        for c in ["offset", "offset_freq"]:
            if c in kwargs:
                setattr(self, c, kwargs.pop(c))
        if len(kwargs) > 0:
            warnings.warn("Arguments {} will not be used".format(kwargs.keys()))
        self.fmin = 0
        self.fmax = STFT.pixel_to_frequency(self, spec.shape[0] - 1)
        # otherwise, a subclass method is called, while fmax (needed by pixel_to_frequency) may not be initialized yet.

    @classmethod
    def from_audio(self, audio):
        dict_ = {"hop_length": HOP_LENGTH, "n_fft": WINDOW_SIZE}
        spec_ = lb.core.stft(y=audio.values, **dict_, window="hann", center=True)
        return STFT(spec_, sr=audio.sr, hop_length=dict_["hop_length"], n_fft=dict_["n_fft"])

    def pixel_to_time(self, p):
        offset = self.offset if hasattr(self, "offset") else 0
        return (self.n_fft / 2 + self.hop_length * p) / self.sr + offset

    def time_to_pixel(self, t):
        offset = self.offset if hasattr(self, "offset") else 0
        return int(max(0, ((t - offset) * self.sr - self.n_fft / 2) / self.hop_length))

    def pixel_to_frequency(self, p, fft_freqs=None):
        if fft_freqs is None:
            fft_freqs = lb.core.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        offset = self.offset_freq if hasattr(self, "offset_freq") else 0
        return fft_freqs[p] + offset

    def frequency_to_pixel(self, f, fft_freqs=None):
        if fft_freqs is None:
            fft_freqs = lb.core.fft_frequencies(self.sr, self.n_fft)

        def find_freq_in_x(f_, x):
            """Index of the last element in x, that is not greater than f_ """
            if f_ > x[-1]:
                return x.shape[0] - 1
            return np.argmax(
                f_ <= x
            )  # if we have all ones, its ok. If its all zeros (ie. big f_), we also get a 0 and hence the if clause above.
            # for ind_, x_ in enumerate(x):
            #    if x_ < f:
            #        return ind_- 1
            # return x.shape[0]

        offset = self.offset_freq if hasattr(self, "offset_freq") else 0
        return find_freq_in_x(f, fft_freqs) - find_freq_in_x(offset, fft_freqs)

    def display(self, **kwargs):
        """ Typical parameters to change for the stft."""
        kwargs.update({c: cv for c, cv in self.__dict__.items() if c != "spec"})
        xcoords = np.array([self.pixel_to_time(p) for p in range(self.spec.shape[1])])
        default_values = {"y_axis": "log", "x_axis": "s", "x_coords": xcoords}
        kwargs.update({c: cv for c, cv in default_values.items() if c not in kwargs})
        fct = partial(dpl.show_spectrogram, **kwargs)
        return dpl.one_plot(fct, arg=self)


class MelSpectrogram(STFT):
    def __init__(self, spec, sr, hop_length, n_fft, fmin, fmax, **kwargs):
        super(MelSpectrogram, self).__init__(spec, sr, hop_length, n_fft, **kwargs)
        if fmax is None:
            fmax = sr / 2
        self.fmin, self.fmax = fmin, fmax

    @classmethod
    def from_audio(self, audio):
        dict_ = {
            "sr": audio.sr,
            "hop_length": HOP_LENGTH,
            "n_fft": WINDOW_SIZE,
            "fmin": F_MIN,
            "fmax": F_MAX,
            "n_mels": NB_BINS,
        }
        spec_ = lb.feature.melspectrogram(
            y=audio.values, **dict_
        )  # , window='hann', center=True, pad_mode='reflect', power=2.0)
        return MelSpectrogram(
            spec_, dict_["sr"], dict_["hop_length"], dict_["n_fft"], dict_["fmin"], dict_["fmax"]
        )

    def pixel_to_frequency(self, p):
        return STFT.pixel_to_frequency(
            self, p, lb.mel_frequencies(n_mels=self.spec.shape[0], fmin=self.fmin, fmax=self.fmax)
        )

    def frequency_to_pixel(self, f):
        # bins = lb.mel_frequencies(n_mels = self.spec.shape[0], fmin = self.fmin, fmax = self.fmax)
        raise NotImplementedError("Test !")
        # return np.argmin(f>bins)
        return STFT.frequency_to_pixel(
            self, f, lb.mel_frequencies(n_mels=self.spec.shape[0], fmin=self.fmin, fmax=self.fmax)
        )

    def display(self, **kwargs):
        """ Typical parameters to change for the mel."""
        xcoords = np.array([self.pixel_to_time(p) for p in range(self.spec.shape[1])])
        kwargs.update({c: cv for c, cv in self.__dict__.items() if c != "spec"})
        default_values = {"y_axis": "mel", "x_axis": "s", "x_coords": xcoords}
        kwargs.update({c: cv for c, cv in default_values.items() if c not in kwargs})
        # dpl.show_spectrogram(self, **kwargs)
        fct = partial(dpl.show_spectrogram, **kwargs)
        return dpl.one_plot(fct, arg=self)
