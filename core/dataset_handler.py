import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


class DatasetHandler:
    batch_size = 64

    def convert_data_to_dataset(self, positive_data, negative_data):
        positive_labels = torch.ones(len(positive_data), dtype=torch.long)
        negative_labels = torch.zeros(len(negative_data), dtype=torch.long)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)

        positive_data_tensor = torch.stack(positive_data)
        negative_data_tensor = torch.stack(negative_data)
        all_data_tensor = torch.cat((positive_data_tensor, negative_data_tensor), dim=0)

        return TensorDataset(all_data_tensor, all_labels)

    def split_dataset_to_data_loaders(self, dataset, count=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert train_ratio + val_ratio + test_ratio == 1, "Ratio sum must be 1"

        if count is not None:
            dataset, _ = random_split(dataset, [count, len(dataset) - count])
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
