from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelAUROC,
)
from torchmetrics import MetricCollection


class ChestNetBase(pl.LightningModule):
    """Base Lightning Module for chest X-ray classification models."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_classes: int = 14,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "accuracy": MultilabelAccuracy(num_labels=num_classes, average="micro"),
                "f1_score": MultilabelF1Score(num_labels=num_classes, average="micro"),
            }
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": MultilabelAccuracy(num_labels=num_classes, average="micro"),
                "f1_score": MultilabelF1Score(num_labels=num_classes, average="micro"),
                "auroc": MultilabelAUROC(num_labels=num_classes),
            }
        )

        self.test_metrics = self.val_metrics.clone()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())

        # Update and log metrics
        metrics = self.train_metrics(y_hat, y)
        self.log_dict(
            {f"train_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())

        # Update and log metrics
        metrics = self.val_metrics(y_hat, y)
        self.log_dict(
            {f"val_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)

        # Update and log metrics
        metrics = self.test_metrics(y_hat, y)
        self.log_dict(
            {f"test_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
        )


class ChestNetS(ChestNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "ChestNetS"
        self.model_details = {
            "architecture": "Small CNN",
            "input_channels": 1,
            "conv_layers": 3,
            "initial_filters": 32,
            "max_filters": 128,
            "dropout_rate": 0.5,
            "final_activation": "sigmoid",
            "num_classes": 14,
        }

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 32x32
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 16x16
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 8x8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),  # This is for 64x64 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 14),
            nn.Sigmoid(),  # Apply sigmoid for multi-label classification or softmax for proper probability distribution
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
