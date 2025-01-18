import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule import ChestDataModule

import structlog

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


def main():
    # Instantiate configs
    train_config = TrainingConfig(num_epochs=50)
    path_config = PathConfig()

    # Create data module
    data_module = ChestDataModule(train_config, path_config)

    # Initialize model
    model = ChestNetS(
        learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    # Configure logger
    mlf_logger = MLFlowLogger(
        tracking_uri=path_config.mlflow_tracking_uri,
        experiment_name="ChestXRayPytorchLightning",
        tags={
            "model_type": model.model_name,
            "dataset": "ChestMNIST",
            "purpose": "production",
            "version": "1.0.0",
            "author": "pgryko",
            "final_activation": "sigmoid",
            "modifications": "Without transforms",
        },
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

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="auto",
        devices=1,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
