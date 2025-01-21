import os

import pytorch_lightning as pl
import torch
from medmnist import ChestMNIST
from torch.utils.data import DataLoader

from src.configs.config import TrainingConfig, PathConfig
from src.data.augmentation import ChestXRayTransforms


class ChestDataModuleLightning(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig, paths: PathConfig):
        super().__init__()
        self.config = config
        self.paths = paths
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def calculate_class_weights(self):
        """Calculate class weights based on training set distribution."""
        # Initialize counters
        num_classes = 14  # For ChestMNIST
        class_counts = torch.zeros(num_classes, dtype=torch.float32)

        # Iterate through training dataset
        for _, labels in DataLoader(self.train_dataset, batch_size=512, num_workers=4):
            class_counts += labels.sum(dim=0)

        # Handle division by zero for classes with no positives
        num_samples = len(self.train_dataset)
        num_negatives = num_samples - class_counts
        pos_weight = num_negatives / (class_counts + 1e-6)  # Add epsilon to avoid NaN

        return pos_weight

    def setup(self, stage: str = None):
        """Setup datasets for each stage of training."""
        if stage == "fit" or stage is None:
            self.train_dataset = ChestMNIST(
                split="train",
                root=self.paths.dataset_root,
                download=True,
                transform=ChestXRayTransforms(
                    is_training=True,
                    rotate_limit=self.config.rotate_limit,
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                ),
                size=64,
            )

            self.val_dataset = ChestMNIST(
                split="val",
                root=self.paths.dataset_root,
                download=True,
                transform=ChestXRayTransforms(is_training=False),
                size=64,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ChestMNIST(
                split="test",
                root=self.paths.dataset_root,
                download=True,
                transform=ChestXRayTransforms(is_training=False),
                size=64,
            )

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

        return data_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,  # For GPU training
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,  # Prefetch data
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
