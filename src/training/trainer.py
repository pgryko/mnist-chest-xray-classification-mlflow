# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score


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
        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=0.0)
        best_val_loss = float("inf")

        with mlflow.start_run():
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

            for epoch in range(self.config.num_epochs):
                train_loss, train_auc = self.train_one_epoch()
                val_loss, val_auc = self.validate_one_epoch()

                # Step the scheduler
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
                else:
                    early_stopping(val_loss)

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Training complete.")
