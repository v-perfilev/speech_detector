import random

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
                            positive_base_path,
                            positive_audio_format,
                            positive_limit_per_folder,
                            negative_base_path,
                            negative_audio_format,
                            negative_limit_per_folder):
        print(f"Data import started")

        positive_spectrograms = self.__create_spectrograms_with_bg(
            positive_base_path,
            positive_audio_format,
            positive_limit_per_folder,
            negative_base_path,
            negative_audio_format
        )

        print(f"Processed {len(positive_spectrograms)} positive spectrograms with background sounds")

        negative_spectrograms = self.__create_spectrograms(
            negative_base_path,
            negative_audio_format,
            negative_limit_per_folder
        )

        print(f"Processed {len(negative_spectrograms)} negative spectrograms")

        dataset = self.dataset_handler.convert_spectrograms_to_dataset(
            positive_spectrograms,
            negative_spectrograms)
        data_loaders = self.dataset_handler.split_dataset_to_data_loaders(dataset)

        print(f"Dataloaders imported")

        return data_loaders

    def __create_spectrograms_with_bg(self, path, audio_format, limit_per_folder, bg_path, bg_audio_format):
        file_paths = self.file_handler.get_file_paths(path, audio_format, limit_per_folder)
        bg_file_paths = self.file_handler.get_file_paths(bg_path, bg_audio_format)

        spectrograms = []

        for file_path in file_paths:
            bg_file_path = random.choice(bg_file_paths)
            samples, rate = self.audio_handler.load_audio(file_path, audio_format)
            bg_samples, _ = self.audio_handler.load_audio(bg_file_path, bg_audio_format)
            mixed_samples = self.audio_handler.mix_audio_samples(samples, bg_samples)
            spectrogram = self.audio_handler.audio_to_spectrogram(mixed_samples, rate)
            spectrograms.append(spectrogram)

        return spectrograms

    def __create_spectrograms(self, path, audio_format, limit_per_folder):
        file_paths = self.file_handler.get_file_paths(path, audio_format, limit_per_folder)

        spectrograms = []

        for file_path in file_paths:
            samples, rate = self.audio_handler.load_audio(file_path, audio_format)
            spectrogram = self.audio_handler.audio_to_spectrogram(samples, rate)
            spectrograms.append(spectrogram)

        return spectrograms
