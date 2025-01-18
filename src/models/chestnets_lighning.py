from typing import Tuple

import mlflow
import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelAUROC,
    MultilabelPrecision,
)
from torchmetrics import MetricCollection
import torchvision.models as models

logger = structlog.get_logger()


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
                "precision": MultilabelPrecision(
                    num_labels=num_classes, average="micro"
                ),
                "f1_score": MultilabelF1Score(num_labels=num_classes, average="micro"),
            }
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": MultilabelAccuracy(num_labels=num_classes, average="micro"),
                "precision": MultilabelPrecision(
                    num_labels=num_classes, average="micro"
                ),
                "f1_score": MultilabelF1Score(num_labels=num_classes, average="micro"),
                # "auroc": MultilabelAUROC(num_labels=num_classes), # very computationally expensive, slow to calculate
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

    # def on_train_start(self):
    #     # Log custom information at start of training
    #     mlflow.log_params(
    #         {
    #             "data_transforms": str(self.trainer.datamodule.transform),
    #             "custom_architecture": "ChestNetS",
    #         }
    #     )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        #  BCEWithLogitsLoss often yields numerical stability and is the recommended pattern for multi-label classification.
        # Use BCEWithLogitsLoss
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        # Update and log metrics
        metrics = self.train_metrics(
            torch.sigmoid(y_hat), y
        )  # apply sigmoid for metrics
        self.log_dict(
            {f"train_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log custom metrics that autolog doesn't capture
        # if self.trainer.is_last_batch:
        #     mlflow.log_metrics({
        #         "custom_metric": some_value,
        #         "batch_specific_metric": another_value
        #     })

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        if batch_idx == 0:  # Only log once
            logger.info(f"Input tensor device: {x.device}")
            logger.info(f"Model device: {next(self.parameters()).device}")
            logger.info(f"Current batch size: {x.shape[0]}")

        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())

        # Update and log metrics
        metrics = self.val_metrics(torch.sigmoid(y_hat), y)

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
        metrics = self.test_metrics(torch.sigmoid(y_hat), y)
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
            "final_activation": "logits",  # changed from 'sigmoid'
            "num_classes": 14,
        }

        # Log custom information that autolog might miss
        # mlflow.log_dict(self.model_details, "model_details.json")

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
            # nn.Sigmoid(),  # Apply sigmoid for multi-label classification or softmax for proper probability distribution
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ChestNetResnet(pl.LightningModule):
    def __init__(self, num_classes=14, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        # Modify the first conv to accept 1 channel
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final layer
        self.backbone.fc = nn.Linear(512, num_classes)
