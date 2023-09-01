from sox import Transformer, Combiner
from . import utils

class MyTransformer(Transformer):
    """ Wrapper around the sox.Transformer class, to first decode (ogg -> wav), apply (wav -> wav) and encode back (wav -> ogg). """

    def apply(self, song_name, input_dir, out_name, output_dir):
        utils.decode(song_name, input_dir, output_dir)
        in_song, out_song =  [output_dir+song_name+'.wav', output_dir + out_name+'.wav']
        self.build(in_song, out_song)
        utils.encode(out_name, output_dir, output_dir)

        utils.remove_wav(output_dir, song_name)
        utils.remove_wav(output_dir, out_name)
        return 0

class NoiseTransformer(MyTransformer):

    def __init__(self, degree, noise_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_type = noise_type
        self.degree = degree

    def apply(self, song_name, input_dir, out_name, output_dir):
        utils.decode(song_name, input_dir, output_dir)
        in_song, out_song, noise_file = [output_dir+song_name+'.wav', output_dir + out_name+'.wav', output_dir+song_name+'noise.wav']
        self.generate_noise(in_song, noise_file, self.noise_type, self.degree)
        Combiner().build([in_song, noise_file], out_song, combine_type = 'mix-power', input_volumes = None)
        utils.encode(out_name, output_dir, output_dir)


        utils.remove_wav(output_dir, song_name)
        utils.remove_wav(output_dir, out_name)
        utils.remove_wav(output_dir, song_name+'noise')
        return 0

    def generate_noise(self, in_song, noise_file, noise_type, degree):
        assert((self.effects is None) or (len(self.effects) == 0))
        extra_args = 'synth {1} vol {0}'.format(degree, noise_type).split( ' ' )
        self.build(in_song, noise_file, extra_args = extra_args)
        return 0

def pitch_shift(song_name, degree, **kwargs):
    t = MyTransformer().pitch(n_semitones = degree)
    return t.apply(song_name = song_name, input_dir = kwargs.get('input_dir', INPUT_DIR), out_name =kwargs.get('out_name', song_name+'obf'), output_dir = kwargs.get('output_dir', OUTPUT_DIR))

