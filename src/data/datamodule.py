from medmnist import ChestMNIST
from torch.utils.data import DataLoader

from src.configs.config import PathConfig, TrainingConfig
from src.data.datamodule_lightning import ChestXRayTransforms


class ChestDataModule:
    def __init__(self, config: TrainingConfig, paths: PathConfig):
        self.config = config
        self.paths = paths

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

        self.test_dataset = ChestMNIST(
            split="test",
            root=self.paths.dataset_root,
            download=True,
            transform=ChestXRayTransforms(is_training=False),
            size=64,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
