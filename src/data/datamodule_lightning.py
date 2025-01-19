import os
from typing import Union

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from PIL.Image import Image
from albumentations.pytorch import ToTensorV2
from medmnist import ChestMNIST
from numpy._typing import NDArray
from torch.utils.data import DataLoader

from src.configs.config import TrainingConfig, PathConfig


class ChestXRayTransforms:
    def __init__(
        self,
        is_training: bool = True,
        rotate_limit: int = 15,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ) -> None:
        if is_training:
            self.transform = A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=rotate_limit,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness, contrast_limit=contrast, p=0.2
                    ),
                    A.Normalize(mean=[0.5], std=[0.5]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0.5], std=[0.5]),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img: Union[Image, NDArray]) -> torch.Tensor:
        # Convert PIL image or numpy array to numpy for albumentations
        return self.transform(image=np.array(img))["image"]


class ChestDataModuleLightning(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig, paths: PathConfig):
        super().__init__()
        self.config = config
        self.paths = paths
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
