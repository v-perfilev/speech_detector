import os
import random

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from core.audio_handler import AudioHandler


def load_dataset(filename="dataset.pt", path="target"):
    return torch.load(path + "/" + filename)


class AudioDataset(Dataset):
    batch_size = 1
    use_mps = False
    use_spectrum = False

    def __init__(self, speech_files, sound_files):
        self.audio_handler = AudioHandler()
        positive_samples = self.__create_mixed_samples(speech_files, sound_files)
        negative_samples = self.__create_samples(sound_files)
        self.data = self.__combine_data(positive_samples, negative_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        data = self.audio_handler.audio_to_spectrum(sample).get_data() \
            if self.use_spectrum else self.audio_handler.audio_to_spectrogram(sample).get_data()
        return data, label

    def save(self, filename="dataset.pt", path="target"):
        os.makedirs(path, exist_ok=True)
        torch.save(self, path + "/" + filename)

    def configure(self, batch_size=32, use_mps=False, use_spectrum=False):
        self.batch_size = batch_size
        self.use_mps = use_mps
        self.use_spectrum = use_spectrum
        return self

    def split_into_data_loaders(self, count=None, train_ratio=0.7, val_ratio=0.15):
        dataset = self

        if count is not None and count < len(dataset):
            dataset, _ = random_split(self, [count, len(dataset) - count])

        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size

        train_dataset, rest_dataset = random_split(dataset, [train_size, dataset_size - train_size])
        val_dataset, test_dataset = random_split(rest_dataset, [val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.__collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.__collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.__collate_fn)

        return train_loader, val_loader, test_loader

    def __create_mixed_samples(self, files, background_files):
        samples = []

        for file in files:
            background_volume = random.choice([i / 10 for i in range(1, 6)])
            background_file = random.choice(background_files)
            sample, _ = self.audio_handler.load_audio(file)
            background_sample, _ = self.audio_handler.load_audio(background_file)
            mixed_sample = self.audio_handler.mix_audio_samples(sample, background_sample, background_volume)

            mixed_sample_chunks = self.audio_handler.divide_audio(mixed_sample.squeeze(0))
            for mixed_sample_chunk in mixed_sample_chunks:
                samples.append(mixed_sample_chunk.unsqueeze(0))

        return samples

    def __create_samples(self, files):
        samples = []

        for file in files:
            sample, _ = self.audio_handler.load_audio(file)

            file_chunks = self.audio_handler.divide_audio(sample.squeeze(0))
            for file_chunk in file_chunks:
                samples.append(file_chunk.unsqueeze(0))

        return samples

    def __combine_data(self, positive_data, negative_data):
        positive_labels = [1] * len(positive_data)
        negative_labels = [0] * len(negative_data)
        all_labels = positive_labels + negative_labels

        all_data = positive_data + negative_data

        return [[data, label] for data, label in zip(all_data, all_labels)]

    def __collate_fn(self, batch):
        device = torch.device("mps" if self.use_mps and torch.backends.mps.is_available() else "cpu")
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = [x.to(device).to(torch.float32) for x in batch]
        return batch
