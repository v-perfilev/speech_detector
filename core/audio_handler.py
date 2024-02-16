import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Resample


class AudioHandler:
    target_sample_rate = None
    target_spectrogram_shape = None
    mix_background_volume = None
    n_mels = 64
    n_fft = 2048
    noise_spectrogram = None

    def __init__(self,
                 target_sample_rate=44100,
                 target_spectrogram_shape=(64, 128),
                 mix_background_volume=0.3):
        self.target_sample_rate = target_sample_rate
        self.target_spectrogram_shape = target_spectrogram_shape
        self.mix_background_volume = mix_background_volume
        self.noise_spectrogram = self.__prepare_noise_spectrogram()

    def load_audio(self, file_path, audio_format):
        if audio_format == 'mp3':
            audio = AudioSegment.from_mp3(file_path)
            samples = torch.tensor(audio.get_array_of_samples()).float()
            rate = audio.frame_rate
        else:
            samples, rate = torchaudio.load(file_path)

        if samples.dim() > 1 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdim=True)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        if rate != self.target_sample_rate:
            resample_transform = Resample(orig_freq=rate, new_freq=self.target_sample_rate)
            samples = resample_transform(samples)
            rate = self.target_sample_rate

        return samples, rate

    def mix_audio_samples(self, main_waveform, background_waveform):
        background_waveform *= self.mix_background_volume

        if main_waveform.shape[1] > background_waveform.shape[1]:
            repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
            background_waveform = background_waveform.repeat(1, repeat_times)
        background_waveform = background_waveform[:, :main_waveform.shape[1]]

        mixed_waveform = main_waveform + background_waveform

        return mixed_waveform

    def audio_to_spectrogram(self, samples, rate):
        spectrogram_transform = MelSpectrogram(rate, n_mels=self.n_mels, n_fft=self.n_fft)
        spectrogram = spectrogram_transform(samples)
        return self.__adjust_spectrogram_shape(spectrogram)

    def prepare_spectrogram(self, spectrogram, reduce_noise=True):
        spectrogram = self.__normalize_spectrogram(spectrogram)
        if reduce_noise:
            spectrogram = self.__reduce_noise(spectrogram)
        return spectrogram

    def is_below_threshold(self, spectrogram, threshold=-35):
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        return True if spectrogram_db.mean() < threshold else False

    def __reduce_noise(self, spectrogram, alpha=1.0):
        assert spectrogram.shape == self.noise_spectrogram.shape, "Spectrograms must have the same shape"
        subtracted_spectrogram = spectrogram - alpha * self.noise_spectrogram
        subtracted_spectrogram = np.maximum(subtracted_spectrogram, 0)
        return subtracted_spectrogram

    def __normalize_spectrogram(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        return (spectrogram - mean) / (std + 1e-6)

    def __adjust_spectrogram_shape(self, spectrogram):
        current_height, current_width = spectrogram.shape[1], spectrogram.shape[2]
        target_height, target_width = self.target_spectrogram_shape

        padding_height = max(0, target_height - current_height)
        padding_width = max(0, target_width - current_width)

        if padding_height > 0 or padding_width > 0:
            padding = [padding_width // 2, padding_width - padding_width // 2,
                       padding_height // 2, padding_height - padding_height // 2]
            spectrogram = F.pad(spectrogram, pad=padding, mode='constant', value=0)

        return spectrogram[:, :target_height, :target_width]

    def __prepare_noise_spectrogram(self):
        noise_audio_path = os.path.abspath("sounds/white_noise.wav")
        noise_audio, noise_rate = self.load_audio(noise_audio_path, 'wav')
        spectrogram = self.audio_to_spectrogram(noise_audio, noise_rate)
        return self.prepare_spectrogram(spectrogram, reduce_noise=False)
