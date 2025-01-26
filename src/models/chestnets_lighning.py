from typing import Tuple, Any, Union, Dict

import mlflow
from mlflow import MlflowException
from torchviz import make_dot

import numpy as np
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
from src.interpretability.explainers import XAIExplainer

logger = structlog.get_logger()


def _calculate_output_shape(layers: nn.Sequential, input_channels: int) -> tuple:
    """
    Calculate the output shape after passing through a sequence of layers.
    """
    # Use 64x64 since that's the expected input size for the model
    dummy_input = torch.zeros(1, input_channels, 64, 64)
    x = dummy_input

    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                x = layer(x)

    return x.shape[1:]


def _infer_input_size(model: nn.Module) -> tuple:
    """
    Infer the required input size for the model based on its architecture.
    """
    # Find first conv layer to get input channels
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    in_channels = first_conv.in_channels

    # Use standard ImageNet input size for ResNet models
    return (1, in_channels, 64, 64)


def log_model_description(
    model: nn.Module,
    config: TrainingConfig,
    input_size: tuple = None,
) -> ModelStatistics:
    """
    Generate and log model summary information using torchinfo and MLflow.

    Args:
        model: The PyTorch model to analyze
        config: Training configuration parameters
        input_size: Optional input tensor size. If None, will be inferred from model architecture.

    Returns:
        ModelStatistics: The generated model statistics
    """
    if input_size is None:
        input_size = _infer_input_size(model)
        logger.info("Inferred input size", input_size=input_size)

    # Validate input size matches model expectations
    if (
        input_size[1]
        != next(m for m in model.modules() if isinstance(m, nn.Conv2d)).in_channels
    ):
        raise ValueError("Input channel mismatch")

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
    try:
        mlflow.log_param("model_name", model.model_name)
        mlflow.log_dict(model.model_details, "model_details.json")
        mlflow.log_text(str(model_stats), "model_summary.txt")
    except MlflowException as e:
        logger.error("Failed to log model params", error=str(e))

    if model_stats:
        try:
            model_params = {
                "total_params": model_stats.total_params,
                "trainable_params": model_stats.trainable_params,
                "non_trainable_params": model_stats.total_params
                - model_stats.trainable_params,
            }
            mlflow.log_params(model_params)
        except MlflowException as e:
            logger.error("Failed to log model params", error=str(e))

    # Log training hyperparameters
    try:
        training_params = {
            "model_name": model.__class__.__name__,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "weight_decay": config.weight_decay,
        }
        mlflow.log_params(training_params)
    except MlflowException as e:
        logger.error("Failed to log model params", error=str(e))

    try:
        sample_input = torch.randn(1, 1, 64, 64).to(next(model.parameters()).device)
        y = model(sample_input)
        dot = make_dot(y, params=dict(model.named_parameters()))

        dot.render("model_architecture", format="png", cleanup=True)
        mlflow.log_artifact("model_architecture.png")
    except MlflowException as e:
        logger.error("Failed to log model params", error=str(e))
    return model_stats


class ChestNetBase(pl.LightningModule):
    """Base Lightning Module for chest X-ray classification models."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_classes: int = 14,
        pos_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.xai_explainer = None
        self.save_hyperparameters()

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        # micro metrics aggregate across all classes before computing the metric, which is often suitable when you have class imbalance.
        # Alternatively, "macro" averages metrics per class and then aggregates, which better highlights per-class performance.

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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        #  BCEWithLogitsLoss often yields numerical stability and is the recommended pattern for multi-label classification.
        # Use BCEWithLogitsLoss
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=self.pos_weight
        )

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
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=self.pos_weight
        )

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

    def setup_interpretability(self):
        """Initialize the XAI explainer after the model is moved to the correct device."""
        if self.xai_explainer is None:
            self.xai_explainer = XAIExplainer(self, self.device)

    def explain_prediction(
        self, x: torch.Tensor, target_class: int, method: str = "gradcam"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate explanation for a prediction.

        Args:
            x: Input tensor
            target_class: Target class to explain
            method: One of ["gradcam", "integrated_gradients", "deep_shap"]

        Returns:
            Explanation (format depends on method)
        """
        self.setup_interpretability()

        if method == "gradcam":
            return self.xai_explainer.compute_gradcam(x, target_class)
        elif method == "integrated_gradients":
            return self.xai_explainer.compute_integrated_gradients(x, target_class)
        elif method == "deep_shap":
            # Need background data for SHAP
            raise NotImplementedError("Need background data for SHAP explanations")
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def get_activation_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get activation maps for each layer.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()

            return hook

        # Register hooks for each layer we want to visualize
        handles = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU)):
                handles.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            self.forward(x)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations


class ChestNetS(ChestNetBase):
    # multi-label classification model, 14 independent binary classifiers
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
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)  # Raw logits
        return x


class ChestNetDebug(ChestNetBase):
    # Used for debugging/testing logging purposes
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "ChestNetDebug"
        self.model_details = {
            "architecture": "Minimal CNN for debugging",
            "input_channels": 1,
            "conv_layers": 1,
            "initial_filters": 16,
            "max_filters": 16,
            "dropout_rate": 0.2,
            "final_activation": "logits",
            "num_classes": 14,
        }

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # Aggressive pooling to reduce dimensions quickly
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16 * 16 * 16, 64),  # Smaller dense layer
            nn.ReLU(inplace=True),
            nn.Linear(64, 14),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ChestNetSAttention(ChestNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "ChestNetS"
        self.model_details = {
            "architecture": "Enhanced CNN with Attention",
            "input_channels": 1,
            "conv_layers": 4,
            "initial_filters": 32,
            "max_filters": 256,
            "dropout_rate": 0.5,
            "final_activation": "none",  # Removed sigmoid as we're using BCEWithLogitsLoss
            "num_classes": 14,
        }

        # Feature Extraction Path
        self.features = nn.ModuleList(
            [
                # Initial Block
                self._make_block(1, 32),
                self._make_block(32, 64),
                self._make_block(64, 128),
                self._make_block(128, 256),
            ]
        )

        # Attention Module
        self.attention = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 14),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction with residual connections
        features = []
        for block in self.features:
            x = block(x)
            features.append(x)

        # Apply attention
        att = self.attention(x)
        x = x * att

        # Classification
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
