import numpy as np
from scipy.special import expit as sigmoid
import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule_lightning import ChestDataModuleLightning

import structlog

from src.interpretability.evaluation import MetricsReporter
from src.models.chestnets_lighning import ChestNetS

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MetricsLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.all_preds = []
        self.all_labels = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called after each test batch.
        `outputs` is whatever your `test_step` returns (e.g. logits, labels)."""
        if outputs is not None:
            logits, labels = outputs
            self.all_preds.append(logits.cpu().numpy())
            self.all_labels.append(labels.cpu().numpy())

    def on_test_end(self, trainer, pl_module):
        """Called after all test batches are processed."""
        all_preds = np.concatenate(self.all_preds, axis=0)
        all_labels = np.concatenate(self.all_labels, axis=0)

        # Convert logits to probability
        y_pred_proba = sigmoid(all_preds)

        reporter = MetricsReporter()
        reporter.calculate_metrics(all_labels, y_pred_proba)
        reporter.log_to_mlflow()

        # Optionally clear the buffers
        self.all_preds = []
        self.all_labels = []


def main():

    # Enable autologging
    mlflow.pytorch.autolog(
        log_every_n_epoch=1,  # Log metrics every epoch
        log_models=True,  # Log model checkpoints
        disable=False,  # Enable autologging
        exclusive=False,  # Allow manual logging alongside autologging
        disable_for_unsupported_versions=False,
        silent=False,  # Print logging info to stdout
    )

    torch.set_float32_matmul_precision("medium")

    # Instantiate configs
    train_config = TrainingConfig(num_epochs=1)
    path_config = PathConfig()

    # Create data module
    data_module = ChestDataModuleLightning(train_config, path_config)

    # Initialize model
    model = ChestNetS(
        learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=train_config.patience, mode="min"
    )

    mlflow.set_tracking_uri(path_config.mlflow_tracking_uri)

    # Let MLFlowLogger Handle the Experiment
    mlf_logger = MLFlowLogger(
        tracking_uri=path_config.mlflow_tracking_uri,
        experiment_name="ChestXRayPytorchLightning",
        run_name="Test post training charts",
        tags={
            "model_type": model.model_name,
            "dataset": "ChestMNIST",
            "purpose": "production",
            "version": "1.0.0",
            "author": "pgryko",
            "final_activation": "sigmoid",
            "modifications": "Without transforms",
        },
        log_model=True,
    )

    mlflow.pytorch.autolog()

    class NoValProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            # Return a disabled progress bar for validation
            bar = super().init_validation_tqdm()
            bar.disable = True
            return bar

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="gpu",
        devices=1,
        logger=mlf_logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            NoValProgressBar(refresh_rate=10),
            MetricsLoggingCallback(),
        ],
        precision=16,
        # profiler="simple",
        enable_model_summary=True,
    )

    # Train model
    trainer.fit(model, data_module)

    # Test the model
    test_results = trainer.test(model, data_module)

    # Log test metrics
    logger.info("Test results", test_results=test_results)

    mlflow.pytorch.log_model(
        model, "model", registered_model_name="chest_xray_classifier"
    )


if __name__ == "__main__":
    main()
