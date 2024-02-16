import random

import matplotlib.pyplot as plt
import torchaudio

from core.audio_handler import AudioHandler
from core.dataset_handler import DatasetHandler
from core.file_handler import FileHandler


class DataMediator:
    audio_handler = None
    dataset_handler = None
    file_handler = None

    def __init__(self, audio_handler=None, dataset_handler=None, file_handler=None):
        self.audio_handler = audio_handler if audio_handler is not None else AudioHandler()
        self.dataset_handler = dataset_handler if dataset_handler is not None else DatasetHandler()
        self.file_handler = file_handler if file_handler is not None else FileHandler()

    def create_data_loaders(self,
                            positive_base_paths,
                            positive_audio_format,
                            positive_limit,
                            negative_base_paths,
                            negative_audio_format,
                            negative_limit):
        print(f"Data import started")

        positive_spectrograms_with_bg = self.__create_spectrograms_with_bg(
            positive_base_paths,
            positive_audio_format,
            int(positive_limit * 0.9),
            negative_base_paths,
            negative_audio_format
        )

        self.draw_random_spectrogram(positive_spectrograms_with_bg)
        print(f"Processed {len(positive_spectrograms_with_bg)} positive spectrograms with background sounds")

        positive_spectrograms_without_bg = self.__create_spectrograms(
            positive_base_paths,
            positive_audio_format,
            int(positive_limit * 0.1)
        )

        self.draw_random_spectrogram(positive_spectrograms_without_bg)
        print(f"Processed {len(positive_spectrograms_without_bg)} positive spectrograms without background sounds")

        negative_spectrograms = self.__create_spectrograms(
            negative_base_paths,
            negative_audio_format,
            negative_limit
        )

        self.draw_random_spectrogram(negative_spectrograms)
        print(f"Processed {len(negative_spectrograms)} negative spectrograms")

        dataset = self.dataset_handler.convert_spectrograms_to_dataset(
            positive_spectrograms_with_bg + positive_spectrograms_without_bg,
            negative_spectrograms)
        data_loaders = self.dataset_handler.split_dataset_to_data_loaders(dataset)

        print(f"Dataloaders imported")

        return data_loaders

    def __create_spectrograms_with_bg(self, paths, audio_format, limit, bg_paths, bg_audio_format):
        self.file_handler.set_total_limit(limit)
        file_paths = self.file_handler.get_file_paths(paths, audio_format)
        self.file_handler.reset_total_limit()
        bg_file_paths = self.file_handler.get_file_paths(bg_paths, bg_audio_format)

        spectrograms = []

        for file_path in file_paths:
            bg_file_path = random.choice(bg_file_paths)
            samples, rate = self.audio_handler.load_audio(file_path, audio_format)
            bg_samples, _ = self.audio_handler.load_audio(bg_file_path, bg_audio_format)
            mixed_samples = self.audio_handler.mix_audio_samples(samples, bg_samples)
            spectrogram = self.audio_handler.audio_to_spectrogram(mixed_samples, rate).prepare()
            spectrograms.append(spectrogram)

        return spectrograms

    def __create_spectrograms(self, paths, audio_format, limit):
        self.file_handler.set_total_limit(limit)
        file_paths = self.file_handler.get_file_paths(paths, audio_format)
        self.file_handler.reset_total_limit()

        spectrograms = []

        for file_path in file_paths:
            samples, rate = self.audio_handler.load_audio(file_path, audio_format)
            spectrogram = self.audio_handler.audio_to_spectrogram(samples, rate).prepare()
            spectrograms.append(spectrogram)

        return spectrograms

    def draw_random_spectrogram(self, spectrograms):
        spectrogram = random.choice(spectrograms)
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db[0].detach().numpy(), cmap='hot', origin='lower', aspect='auto')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()
