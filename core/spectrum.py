import torch
import torch.nn.functional as F
import torchaudio.transforms as T


class Spectrum:
    spectrum = None
    spectrum_size = None
    noise_spectrum = None
    n_fft = 2048

    def __init__(self, samples, spectrogram_size=1025, noise_spectrum=None):
        fft_result = torch.fft.fft(samples, n=self.n_fft)
        self.spectrum = torch.abs(fft_result[:self.n_fft // 2])
        self.spectrum_size = spectrogram_size
        self.noise_spectrum = noise_spectrum

    def is_below_threshold(self, threshold=-40):
        spectrum_db = T.AmplitudeToDB(stype='magnitude')(self.spectrum.unsqueeze(0).unsqueeze(0))
        mean_db = spectrum_db.mean().item()
        return True if mean_db < threshold else False

    def prepare(self):
        self.__adjust_form()
        if self.noise_spectrum is not None:
            self.__reduce_noise()
            self.__normalize()
        return self.spectrum

    def __normalize(self):
        mean = self.spectrum.mean()
        std = self.spectrum.std()
        self.spectrum = (self.spectrum - mean) / (std + 1e-6)

    def __adjust_form(self):
        current_size = self.spectrum.shape[0]
        if current_size > self.spectrum_size:
            self.spectrum = self.spectrum[:self.spectrum_size]
        elif current_size < self.spectrum_size:
            padding_size = self.spectrum_size - current_size
            self.spectrum = F.pad(self.spectrum, (0, padding_size), "constant", 0)

    def __reduce_noise(self, alpha=1.0):
        assert self.spectrum.shape == self.noise_spectrum.shape, "Spectrum and noise spectrum must have the same shape"
        subtracted_spectrum = self.spectrum - alpha * self.noise_spectrum
        subtracted_spectrum = torch.clamp(subtracted_spectrum, min=0)
        self.spectrum = subtracted_spectrum
