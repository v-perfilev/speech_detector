import os

import torch
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import Resample

from core.spectrogram import Spectrogram
from core.spectrum import Spectrum


class AudioHandler:
    target_sample_rate = None
    target_spectrum_size = None
    target_spectrogram_shape = None
    mix_background_volume = None
    noise_spectrum = None
    noise_spectrogram = None

    def __init__(self,
                 target_sample_rate=44100,
                 target_spectrum_size=1025,
                 target_spectrogram_shape=(64, 256),
                 mix_background_volume=0.4):
        self.target_sample_rate = target_sample_rate
        self.target_spectrum_size = target_spectrum_size
        self.target_spectrogram_shape = target_spectrogram_shape
        self.mix_background_volume = mix_background_volume
        self.noise_spectrum = self.__prepare_noise_spectrum()
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

    def audio_to_spectrum(self, samples):
        return Spectrum(samples, self.target_spectrum_size, self.noise_spectrum)

    def audio_to_spectrogram(self, samples, rate):
        return Spectrogram(samples, rate, self.target_spectrogram_shape, self.noise_spectrogram)

    def __prepare_noise_spectrum(self):
        noise_audio_path = os.path.abspath("sounds/white_noise.wav")
        noise_audio, noise_rate = self.load_audio(noise_audio_path, 'wav')
        return Spectrum(noise_audio).prepare()

    def __prepare_noise_spectrogram(self):
        noise_audio_path = os.path.abspath("sounds/white_noise.wav")
        noise_audio, noise_rate = self.load_audio(noise_audio_path, 'wav')
        return Spectrogram(noise_audio, noise_rate).prepare()
