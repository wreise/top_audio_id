import librosa as lb
import soundfile as sf

import IPython.display as ipd

DEFAULT_SR = 44100 #22050
DEFAULT_MONO = True

class Audio(object):

    def __init__(self, samples, sr, mono):
        self.values = samples
        self.sr = sr
        self.mono = mono

    @classmethod
    def from_file(self, file_name, sr = DEFAULT_SR, mono = DEFAULT_MONO):
        song, sr_ = lb.core.load(file_name, sr = sr, mono = mono)
        return Audio(samples = song,
                     sr = sr, mono = mono)

    def sample_to_time(self, n):
        return n/self.sr

    def time_to_sample(self, t):
        """ Convert the time, in seconds, to a sample index (the one before).
        Params:
            t: float, Time in seconds
        Returns:
            index, (int) positive"""
        assert t>=0, "The time 't' must be >=0"
        return int(t*self.sr)

    def display(self):
        return ipd.Audio(data = self.values, rate = self.sr)

    def extract(self, tStart, tEnd):
        """ Define another audio, by extracting a range from the current audio
        Params:
            tStart, tEnd: float, times in seconds, defining the range to keep"""
        assert (tStart<tEnd), """The start time must be smaller than the end time.""" #(self.time_to_sample(tEnd - tStart)< self.values.shape[0]) &
        t0, t1 = [self.time_to_sample(c) for c in [tStart, tEnd]]
        assert (t1< self.values.shape[0]), """The desired interval does not fit in the audio."""
        return Audio(self.values.copy()[t0:t1], self.sr, self.mono)

    def to_file(self, file_name):
        sf.write(file_name, data = self.values, samplerate = self.sr)
        import warnings
        warnings.warn("Test!")
        return True
