import mlflow
import torch
from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule import ChestDataModule
from src.models.chestnets import ChestNetS
from src.training.trainer import ChestXRayTrainer
from src.interpretability.evaluation import MetricsReporter, evaluate_model

import structlog

# Configure structlog
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
    train_config = TrainingConfig(num_epochs=5)
    path_config = PathConfig()

    # Prepare device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Using device", device=device)

    # DataModule
    data_module = ChestDataModule(train_config, path_config)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Choose model
    model = ChestNetS().to(device)

    # Trainer
    trainer = ChestXRayTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        mlflow_tracking_uri=path_config.mlflow_tracking_uri,
        experiment_tags={
            "model_type": "ChestNetS",
            "dataset": "ChestMNIST",
            "purpose": "production",
            "version": "1.0.0",
            "author": "pgryko",
            "final_activation": "softmax",
        },
        experiment_description="""
    Training run for chest X-ray classification using ChestNetS architecture.
    Dataset: ChestMNIST
    Purpose: Production model training
    Special notes: Using augmented dataset with enhanced preprocessing
    """,
    )

    run_id = trainer.train_model()

    # Evaluate on the test set
    y_true, y_prob = evaluate_model(model, test_loader, device)

    # Generate final metrics using same run
    with mlflow.start_run(run_id=run_id):
        reporter = MetricsReporter()
        reporter.calculate_metrics(y_true, y_prob)
        reporter.log_to_mlflow()

        logger.info("Test metrics:", reporter.metrics)


if __name__ == "__main__":
    main()
