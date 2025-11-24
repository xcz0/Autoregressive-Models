# SequentialData.py

import torch
from torch.utils.data import Dataset

from lightning.pytorch import LightningDataModule


class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length=1, stride=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self) -> int:
        return self.data.size(0) - self.sequence_length // self.stride

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx * self.stride : idx * self.stride + self.sequence_length]
        y = self.data[idx * self.stride + self.sequence_length]
        return x, y


class SequenceDataModule(LightningDataModule):
    def __init__(
        self, sequence_length=10, stride=1, batch_size=32, ration=(0.7, 0.15, 0.15)
    ):
        """
        docstring
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Example data, replace with actual data loading logic
        data = torch.arange(100).float()

        # Split data into train, val, test sets
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        test_size = len(data) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            data, [train_size, val_size, test_size]
        )

        self.train_dataset = SequenceDataset(
            train_data, self.sequence_length, self.stride
        )
        self.val_dataset = SequenceDataset(val_data, self.sequence_length, self.stride)
        self.test_dataset = SequenceDataset(
            test_data, self.sequence_length, self.stride
        )
