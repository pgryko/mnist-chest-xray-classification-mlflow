# src/training/trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import structlog
from mlflow.models.signature import infer_signature

from torchinfo import summary, ModelStatistics


logger = structlog.get_logger()


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ChestXRayTrainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        config,
        mlflow_tracking_uri=None,
        experiment_tags=None,
        experiment_description=None,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Set up optimizer, scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        self.experiment_tags = experiment_tags or {}
        self.experiment_description = experiment_description

        # MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def log_model_summary(self):
        # Use torchinfo instead of torchsummary as its newer and supports mac GPUS
        model_stats: ModelStatistics = summary(
            self.model,
            input_size=(1, 1, 64, 64),  # (batch_size, channels, height, width)
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
        mlflow.log_param("model_name", self.model.model_name)
        mlflow.log_dict(self.model.model_details, "model_details.json")
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
            "model_name": self.model.__class__.__name__,
            "lr": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "weight_decay": self.config.weight_decay,
        }
        mlflow.log_params(training_params)

    def _calculate_metrics(self, targets_list, preds_list):
        # Convert predictions to binary (0/1) using 0.5 threshold
        preds_binary = (np.array(preds_list) > 0.5).astype(int)
        targets = np.array(targets_list)

        return {
            "accuracy": accuracy_score(targets, preds_binary),
            "precision": precision_score(targets, preds_binary),
            "recall": recall_score(targets, preds_binary),
            "f1": f1_score(targets, preds_binary),
            "roc_auc": roc_auc_score(targets, preds_list),
            "confusion_matrix": confusion_matrix(targets, preds_binary).tolist(),
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        targets_list = []
        preds_list = []

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.float().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

            # Collect stats
            targets_list.extend(target.cpu().numpy())
            preds_list.extend(output.detach().cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(targets_list, preds_list)
        return avg_loss, metrics

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        targets_list = []
        preds_list = []

        with torch.no_grad():
            for data, target in self.val_loader:
                if data.shape[-1] != 64:
                    logger.warning("Resizing image", shape=data.shape)
                    data = nn.functional.interpolate(
                        data, size=(64, 64), mode="bilinear"
                    )

                data, target = data.to(self.device), target.float().to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                targets_list.extend(target.cpu().numpy())
                preds_list.extend(output.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(targets_list, preds_list)
        return avg_loss, metrics

    def train_model(self, experiment_name="ChestXRay"):

        if experiment_name:
            mlflow.set_experiment(experiment_name)
            # Set experiment description if provided
            if self.experiment_description:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    mlflow.set_experiment_tag(
                        "mlflow.note.content",
                        self.experiment_description,
                    )

        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=0.0)
        best_val_loss = float("inf")

        with mlflow.start_run(log_system_metrics=True):
            # Log tags
            mlflow.set_tags(self.experiment_tags)

            # Log initial hyperparams
            self.log_model_summary()

            for epoch in range(self.config.num_epochs):
                train_loss, train_metrics = self.train_one_epoch()
                val_loss, val_metrics = self.validate_one_epoch()

                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Log metrics to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics["accuracy"],
                        "train_precision": train_metrics["precision"],
                        "train_recall": train_metrics["recall"],
                        "train_f1": train_metrics["f1"],
                        "train_auc": train_metrics["roc_auc"],
                        "val_loss": val_loss,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1"],
                        "val_auc": val_metrics["roc_auc"],
                    },
                    step=epoch,
                )

                # Log confusion matrix as a separate artifact
                mlflow.log_dict(
                    {
                        "train_confusion_matrix": train_metrics["confusion_matrix"],
                        "val_confusion_matrix": val_metrics["confusion_matrix"],
                    },
                    f"confusion_matrices_epoch_{epoch}.json",
                )

                logger.info(
                    "Training epoch completed",
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs,
                    train_loss=round(train_loss, 4),
                    train_metrics={
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in train_metrics.items()
                    },
                    val_loss=round(val_loss, 4),
                    val_metrics={
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in val_metrics.items()
                    },
                )

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.pytorch.log_model(
                        self.model,
                        artifact_path="best_model",
                        signature=infer_signature(
                            next(iter(self.val_loader))[0].numpy(),
                            next(iter(self.val_loader))[1].numpy(),
                        ),
                    )
                    early_stopping.counter = 0

                else:
                    early_stopping(val_loss)

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Training complete.")
