import numpy as np
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram


class Spectrogram:
    spectrogram = None
    spectrogram_shape = None
    noise_spectrogram = None
    n_mels = 64
    n_fft = 2048
    hop_length = 512

    def __init__(self, samples, rate, spectrogram_shape=(64, 256), noise_spectrogram=None):
        spectrogram_transform = MelSpectrogram(rate, hop_length=self.hop_length, n_mels=self.n_mels, n_fft=self.n_fft)
        self.spectrogram = spectrogram_transform(samples)
        self.spectrogram_shape = spectrogram_shape
        self.noise_spectrogram = noise_spectrogram

    def is_below_threshold(self, threshold=-35):
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(self.spectrogram)
        return True if spectrogram_db.mean() < threshold else False

    def prepare(self):
        self.__adjust_form()
        if self.noise_spectrogram is not None:
            self.__reduce_noise()
            self.__normalize()
        return self.spectrogram

    def __normalize(self):
        mean = self.spectrogram.mean()
        std = self.spectrogram.std()
        self.spectrogram = (self.spectrogram - mean) / (std + 1e-6)

    def __adjust_form(self):
        current_height, current_width = self.spectrogram.shape[1], self.spectrogram.shape[2]
        target_height, target_width = self.spectrogram_shape

        padding_height = max(0, target_height - current_height)
        padding_width = max(0, target_width - current_width)

        if padding_height > 0 or padding_width > 0:
            padding = [padding_width // 2, padding_width - padding_width // 2,
                       padding_height // 2, padding_height - padding_height // 2]
            self.spectrogram = F.pad(self.spectrogram, pad=padding, mode='constant', value=0)

        self.spectrogram = self.spectrogram[:, :target_height, :target_width]

    def __reduce_noise(self, alpha=1.0):
        assert self.spectrogram.shape == self.noise_spectrogram.shape, "Spectrograms must have the same shape"
        subtracted_spectrogram = self.spectrogram - alpha * self.noise_spectrogram
        subtracted_spectrogram = np.maximum(subtracted_spectrogram, 0)
        self.spectrogram = subtracted_spectrogram
