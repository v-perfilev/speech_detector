import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.abstract_mediator import AbstractMediator


class SpectrumMediator(AbstractMediator):

    def __init__(self, audio_handler=None, dataset_handler=None, file_handler=None):
        super().__init__(audio_handler, dataset_handler, file_handler)

    def prepare_example(self, samples, rate):
        return self.audio_handler.audio_to_spectrum(samples).prepare()

    def draw_random_example(self, data):
        example = random.choice(data)

        spectrum = example.squeeze().detach().numpy()
        rate = self.audio_handler.target_sample_rate
        frequencies = np.linspace(0, rate / 2, len(spectrum))

        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Spectrum')
        plt.tight_layout()
        plt.show()

    def save_dataset(self, dataset):
        os.makedirs('tmp', exist_ok=True)
        torch.save(dataset, 'tmp/spectrum_dataset.pt')

    def load_dataset(self, count=None):
        dataset = torch.load('tmp/spectrum_dataset.pt')
        return self.dataset_handler.split_dataset_to_data_loaders(dataset, count=count)
