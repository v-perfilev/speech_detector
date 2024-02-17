import os
import random

import matplotlib.pyplot as plt
import torch
import torchaudio

from core.abstract_mediator import AbstractMediator


class SpectrogramMediator(AbstractMediator):

    def prepare_example(self, samples, rate):
        return self.audio_handler.audio_to_spectrogram(samples, rate).prepare()

    def __init__(self, audio_handler=None, dataset_handler=None, file_handler=None):
        super().__init__(audio_handler, dataset_handler, file_handler)

    def draw_random_example(self, data):
        example = random.choice(data)

        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(example)

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db[0].detach().numpy(), cmap='hot', origin='lower', aspect='auto')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()

    def save_dataset(self, dataset):
        os.makedirs('tmp', exist_ok=True)
        torch.save(dataset, 'tmp/spectrogram_dataset.pt')

    def load_dataset(self, count=None):
        dataset = torch.load('tmp/spectrogram_dataset.pt')
        return self.dataset_handler.split_dataset_to_data_loaders(dataset, count=count)
