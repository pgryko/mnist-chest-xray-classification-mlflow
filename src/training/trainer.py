# src/training/trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score
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
        self, model, device, train_loader, val_loader, config, mlflow_tracking_uri=None
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

        # MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
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

            # Log additional model statistics
            if model_stats:
                mlflow.log_params(
                    {
                        "total_params": model_stats.total_params,
                        "trainable_params": model_stats.trainable_params,
                        "non_trainable_params": model_stats.total_params
                        - model_stats.trainable_params,
                    }
                )

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
        roc_auc = roc_auc_score(targets_list, preds_list)
        return avg_loss, roc_auc

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
        roc_auc = roc_auc_score(targets_list, preds_list)
        return avg_loss, roc_auc

    def train_model(self):
        # mlflow.end_run()

        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=0.0)
        best_val_loss = float("inf")

        with mlflow.start_run(log_system_metrics=True):
            # Log initial hyperparams
            mlflow.log_params(
                {
                    "model_name": self.model.__class__.__name__,
                    "lr": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "weight_decay": self.config.weight_decay,
                }
            )

            # Create an input example
            sample_input = torch.randn(
                1, 1, 64, 64
            )  # Batch size 1, 1 channel, 64x64 image

            # Generate model output for the sample input
            self.model.eval()
            with torch.no_grad():
                sample_output = self.model(sample_input)

            # Infer the signature from sample input and output
            signature = infer_signature(
                sample_input.numpy(),  # Convert to numpy for signature
                sample_output.numpy(),  # Convert to numpy for signature
            )

            # Log the model with signature and input example
            mlflow.pytorch.log_model(
                self.model,
                "model",
                signature=signature,
                input_example=sample_input.numpy(),
            )

            for epoch in range(self.config.num_epochs):
                train_loss, train_auc = self.train_one_epoch()
                val_loss, val_auc = self.validate_one_epoch()

                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Log metrics to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_auc": train_auc,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                    },
                    step=epoch,
                )

                print(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} "
                    f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.pytorch.log_model(self.model, artifact_path="best_model")
                    early_stopping.counter = 0

                    # Collect predictions from best validation epoch
                    self.model.eval()
                    final_targets = []
                    final_preds = []
                    with torch.no_grad():
                        for data, target in self.val_loader:
                            if data.shape[-1] != 64:
                                data = nn.functional.interpolate(
                                    data, size=(64, 64), mode="bilinear"
                                )
                            data, target = data.to(self.device), target.float().to(
                                self.device
                            )
                            output = self.model(data)
                            final_targets.extend(target.cpu().numpy())
                            final_preds.extend(output.cpu().numpy())
                else:
                    early_stopping(val_loss)

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Training complete.")
        return np.array(final_targets), np.array(final_preds)
