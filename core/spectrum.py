import torch
import torch.nn.functional as F
import torchaudio.transforms as T


class Spectrum:

    def __init__(self, sample, target_size, n_fft):
        fft_result = torch.fft.fft(sample, n=n_fft)
        self.data = torch.abs(fft_result[:n_fft // 2])
        self.target_size = target_size

    def is_below_threshold(self, threshold=-40):
        spectrum_db = T.AmplitudeToDB(stype='magnitude')(self.data.unsqueeze(0).unsqueeze(0))
        mean_db = spectrum_db.mean().item()
        return True if mean_db < threshold else False

    def get_data(self):
        self.__adjust_form()
        return self.data

    def __adjust_form(self):
        current_size = self.data.shape[0]
        if current_size > self.target_size:
            self.data = self.data[:self.target_size]
        elif current_size < self.target_size:
            padding_size = self.target_size - current_size
            self.data = F.pad(self.data, (0, padding_size), "constant", 0)
