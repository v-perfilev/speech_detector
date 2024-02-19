import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram


class Spectrogram:

    def __init__(self, sample, rate, target_shape, n_mels, n_fft, hop_length):
        spectrogram_transform = MelSpectrogram(rate, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)
        self.data = spectrogram_transform(sample)
        self.target_shape = target_shape

    def is_below_threshold(self, threshold=-35):
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(self.data)
        return True if spectrogram_db.mean() < threshold else False

    def get_data(self):
        self.__adjust_form()
        return self.data

    def __adjust_form(self):
        current_height, current_width = self.data.shape[1], self.data.shape[2]
        target_height, target_width = self.target_shape

        padding_height = max(0, target_height - current_height)
        padding_width = max(0, target_width - current_width)

        if padding_height > 0 or padding_width > 0:
            padding = [padding_width // 2, padding_width - padding_width // 2,
                       padding_height // 2, padding_height - padding_height // 2]
            self.data = F.pad(self.data, pad=padding, mode='constant', value=0)

        self.data = self.data[:, :target_height, :target_width]
