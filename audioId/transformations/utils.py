from functools import partial
import subprocess

def encode_command(input_filename, output_filename):
    #subprocess.call('oggdec -q 6 {0} -o {1}'.format(input_filename, output_filename))
    f = 'ffmpeg -i {0} {1}'.format(input_filename, output_filename).split(' ')
    return f

def complete_command(fct, song_name, input_format, output_format, input_dir, output_dir):
    return fct(input_filename=input_dir+song_name+input_format,
                          output_filename=output_dir+ song_name+ output_format)

def decode_command(input_filename, output_filename):
    f = 'oggenc -q 6 {0} -o {1}'.format(input_filename, output_filename).split(' ')
    return f

def remove_wav(dir_name, song_name):
    file_name = dir_name+song_name+'.wav'
    try:
        execute_command(['rm', file_name])
    except:
        print('Could not remove {0}'.format(file_name))
        return 1
    return 0

def execute_command(command):
    subprocess.check_output(command)
    return 0

def decode(song_name, input_dir, output_dir):
    is_format_in_song_name = len(song_name.split("."))>1
    if is_format_in_song_name:
        input_format = song_name.split(".")
    else:
        input_format = ".ogg"
    return execute_command(command = complete_command(encode_command, song_name,
                                                                      input_format = input_format, output_format ='.wav', input_dir = input_dir, output_dir = output_dir))
#decode = lambda song_name, input_dir, output_dir: execute_command(command = complete_command(encode_command, song_name,
#                                                                      input_format = '.ogg', output_format ='.wav', input_dir = input_dir, output_dir = output_dir))
encode = lambda song_name, input_dir, output_dir: execute_command(command = complete_command(decode_command, song_name,
                                                                      input_format = '.wav', output_format = '.ogg', input_dir = input_dir, output_dir = output_dir))
