import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class DatasetHandler:
    batch_size = 32

    def convert_spectrograms_to_dataset(self, positive_spectrograms, negative_spectrograms):
        speech_labels = torch.ones(len(positive_spectrograms), dtype=torch.long)
        sounds_labels = torch.zeros(len(negative_spectrograms), dtype=torch.long)
        all_labels = torch.cat((speech_labels, sounds_labels), dim=0)

        speech_spectrograms_tensor = torch.stack(positive_spectrograms)
        sounds_spectrograms_tensor = torch.stack(negative_spectrograms)
        all_spectrograms_tensor = torch.cat((speech_spectrograms_tensor, sounds_spectrograms_tensor), dim=0)

        return TensorDataset(all_spectrograms_tensor, all_labels)

    def split_dataset_to_data_loaders(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert train_ratio + val_ratio + test_ratio == 1, "Ratio sum must be 1"

        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size

        train_dataset, rest_dataset = random_split(dataset, [train_size, dataset_size - train_size])
        val_dataset, test_dataset = random_split(rest_dataset, [val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
