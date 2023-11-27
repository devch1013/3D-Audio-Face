import os

from scipy.io.wavfile import write


def save_wav(audio, rate, filename, output_path):
    os.makedirs(output_path, exist_ok=True)
    filename = filename + '.wav'
    save_path = os.path.join(output_path, filename)
    write(data=audio, rate=rate, filename=save_path)