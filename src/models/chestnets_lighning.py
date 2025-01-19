from typing import Tuple, Any

import mlflow
from torchviz import make_dot

import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
)
from torchmetrics import MetricCollection

from src.configs.config import TrainingConfig

logger = structlog.get_logger()


def log_model_summary(
    model: nn.Module,
    config: TrainingConfig,
    input_size: tuple = (1, 1, 64, 64),
) -> ModelStatistics:
    """
    Generate and log model summary information using torchinfo and MLflow.

    Args:
        model: The PyTorch model to analyze
        config: Training configuration parameters
        logger: Logger instance for logging information
        input_size: Input tensor size for model summary (batch_size, channels, height, width)

    Returns:
        ModelStatistics: The generated model statistics
    """
    # Generate model summary
    model_stats: ModelStatistics = summary(
        model,
        input_size=input_size,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        col_width=20,
        row_settings=["var_names"],
        verbose=True,
    )

    logger.info("Model Summary", model_stats=model_stats)

    # Log model details and summary
    mlflow.log_param("model_name", model.model_name)
    mlflow.log_dict(model.model_details, "model_details.json")
    mlflow.log_text(str(model_stats), "model_summary.txt")

    if model_stats:
        model_params = {
            "total_params": model_stats.total_params,
            "trainable_params": model_stats.trainable_params,
            "non_trainable_params": model_stats.total_params
            - model_stats.trainable_params,
        }
        mlflow.log_params(model_params)

    # Log training hyperparameters
    training_params = {
        "model_name": model.__class__.__name__,
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
    }
    mlflow.log_params(training_params)

    sample_input = torch.randn(1, 1, 64, 64).to(next(model.parameters()).device)
    y = model(sample_input)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    mlflow.log_artifact("model_architecture.png")

    return model_stats


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

    def on_train_start(self):
        # Log custom information at start of training
        # mlflow.log_params(
        #     {
        #         "data_transforms": str(self.trainer.datamodule.transform),
        #         "custom_architecture": "ChestNetS",
        #     }
        # )

        print("on_train_start")

        # log_model_summary(self, self.hparams)

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
        #     mlflow.log_metrics(
        #         {"custom_metric": some_value, "batch_specific_metric": another_value}
        #     )

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
    ) -> dict[str, Tensor | Any]:
        x, y = batch
        y_hat = self(x)

        # Update and log metrics
        metrics = self.test_metrics(torch.sigmoid(y_hat), y)
        self.log_dict(
            {f"test_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
        )
        return {"logits": y_hat, "labels": y}


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
