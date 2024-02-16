import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from scipy.signal import butter, sosfiltfilt
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Resample


class AudioHandler:
    target_sample_rate = 44100
    target_spectrogram_shape = (64, 128)
    mix_background_volume = 0.3
    detection_threshold = -40
    n_mels = 64
    n_fft = 2048

    def __init__(self,
                 target_sample_rate=44100,
                 target_spectrogram_shape=(64, 128),
                 mix_bg_volume=0.3,
                 detection_threshold=-40):
        self.target_sample_rate = target_sample_rate
        self.target_spectrogram_shape = target_spectrogram_shape
        self.mix_background_volume = mix_bg_volume
        self.detection_threshold = detection_threshold

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

        samples = self.__apply_bandpass_filter(samples)

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
        spectrogram = self.__normalize_spectrogram(spectrogram)
        spectrogram = self.__adjust_spectrogram_shape(spectrogram)
        return spectrogram

    def is_below_threshold(self, spectrogram):
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        return True if spectrogram_db.mean() < self.detection_threshold else False

    def __apply_bandpass_filter(self, samples, lowcut=300, highcut=3400, order=5):
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=self.target_sample_rate, output='sos')
        filtered_samples = sosfiltfilt(sos, samples.numpy())
        return torch.tensor(filtered_samples, dtype=torch.float)

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
