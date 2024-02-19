import torch
import torchaudio
from torchaudio.transforms import Resample

from core.spectrogram import Spectrogram
from core.spectrum import Spectrum


class AudioHandler:
    n_mels = 64
    n_fft = 2048
    hop_length = 512
    spectrum_size = 1025
    spectrogram_shape = (64, 256)

    def __init__(self, sample_rate=44100, chunk_size=14000):
        self.target_sample_rate = sample_rate
        self.chunk_size = chunk_size

    def load_audio(self, file_path):
        samples, rate = torchaudio.load(file_path)
        return self.prepare_audio(samples, rate)

    def prepare_audio(self, samples, rate):
        if samples.dim() > 1 and samples.shape[0] == 2:
            samples = samples.mean(dim=0, keepdim=True)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        if rate != self.target_sample_rate:
            resample_transform = Resample(orig_freq=rate, new_freq=self.target_sample_rate)
            samples = resample_transform(samples)
            rate = self.target_sample_rate

        return samples, rate

    def mix_audio_samples(self, main_waveform, background_waveform, background_volume):
        background_waveform *= background_volume

        if main_waveform.shape[1] > background_waveform.shape[1]:
            repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
            background_waveform = background_waveform.repeat(1, repeat_times)
        background_waveform = background_waveform[:, :main_waveform.shape[1]]

        mixed_waveform = main_waveform + background_waveform

        return mixed_waveform

    def divide_audio(self, audio):
        if self.chunk_size > audio.size(0):
            return []
        chunks = audio.unfold(0, self.chunk_size, self.chunk_size).contiguous()
        processed_chunks = []
        for chunk in chunks:
            if chunk.size(0) < self.chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_size - chunk.size(0)))
            processed_chunks.append(chunk)
        return processed_chunks

    def audio_to_spectrum(self, samples):
        return Spectrum(samples, target_size=self.spectrum_size, n_fft=self.n_fft)

    def audio_to_spectrogram(self, samples):
        return Spectrogram(samples, self.target_sample_rate, target_shape=self.spectrogram_shape,
                           n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)

    def save_audio(self, audio_data, filename, path="target"):
        torchaudio.save(path + "/" + filename, audio_data, self.target_sample_rate)
