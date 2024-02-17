import random
from abc import abstractmethod


class AbstractMediator:
    audio_handler = None
    dataset_handler = None
    file_handler = None

    def __init__(self, audio_handler, dataset_handler, file_handler):
        self.audio_handler = audio_handler
        self.dataset_handler = dataset_handler
        self.file_handler = file_handler

    def create_dataset(self,
                       positive_base_paths,
                       positive_audio_format,
                       positive_limit,
                       negative_base_paths,
                       negative_audio_format,
                       negative_limit):
        print(f"Data import started")

        positive_data = self.__create_mixed_audio_data(
            positive_base_paths,
            positive_audio_format,
            positive_limit,
            negative_base_paths,
            negative_audio_format
        )

        self.draw_random_example(positive_data)
        print(f"Processed {len(positive_data)} positive data")

        negative_data = self.__create_audio_data(
            negative_base_paths,
            negative_audio_format,
            negative_limit
        )

        self.draw_random_example(negative_data)
        print(f"Processed {len(negative_data)} negative data")

        return self.dataset_handler.convert_data_to_dataset(positive_data, negative_data)

    def __create_mixed_audio_data(self, positive_paths, audio_format, limit, negative_paths, negative_audio_format):
        self.file_handler.set_total_limit(limit)
        positive_file_paths = self.file_handler.get_file_paths(positive_paths, audio_format)
        self.file_handler.reset_total_limit()

        negative_file_paths = self.file_handler.get_file_paths(negative_paths, negative_audio_format)

        data = []

        for positive_file_path in positive_file_paths:
            negative_file_path = random.choice(negative_file_paths)
            positive_samples, rate = self.audio_handler.load_audio(positive_file_path, audio_format)
            negative_samples, _ = self.audio_handler.load_audio(negative_file_path, negative_audio_format)
            mixed_samples = self.audio_handler.mix_audio_samples(positive_samples, negative_samples)
            example = self.prepare_example(mixed_samples, rate)
            data.append(example)

        return data

    def __create_audio_data(self, paths, audio_format, limit):
        self.file_handler.set_total_limit(limit)
        file_paths = self.file_handler.get_file_paths(paths, audio_format)
        self.file_handler.reset_total_limit()

        data = []

        for file_path in file_paths:
            samples, rate = self.audio_handler.load_audio(file_path, audio_format)
            example = self.prepare_example(samples, rate)
            data.append(example)

        return data

    @abstractmethod
    def prepare_example(self, samples, rate):
        pass

    @abstractmethod
    def draw_random_example(self, data):
        pass

    @abstractmethod
    def save_dataset(self, dataset):
        pass

    @abstractmethod
    def load_dataset(self):
        pass
