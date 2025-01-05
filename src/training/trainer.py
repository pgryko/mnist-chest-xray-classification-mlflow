# src/training/trainer.py
import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchinfo import ModelStatistics, summary

logger = structlog.get_logger()


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: float | None = None
        self.early_stop: bool = False

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
        model: nn.Module,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: any,
        mlflow_tracking_uri: str | None = None,
        experiment_tags: dict[str, str] | None = None,
        experiment_description: str | None = None,
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

    """
    Metrics

    Accuracy: Gives an overall picture of correct classifications, but may be misleading if classes are imbalanced.

    Precision: Shows how many of the positive predictions were actually correct. In medical contexts, this indicates the reliability of positive diagnoses.

    Recall: Shows how many actual positive cases were correctly identified. In medical contexts, this indicates how many actual cases of the condition were caught.

    F1-score: Provides a balanced measure between precision and recall, which is especially useful when you need to find an optimal balance between false positives and false negatives.

    Confusion Matrix: Gives a detailed breakdown of correct and incorrect predictions for each class, which is crucial for understanding the model's error patterns.

    This enhanced metrics logging will provide a more comprehensive view of the model's performance, especially important in medical applications where understanding different types of errors is crucial. The metrics can be viewed in the MLflow UI and used to make informed decisions about model selection and threshold tuning.

    For this specific chest X-ray classification task, these metrics are particularly relevant as they can help:

    Evaluate if the model is biased towards certain predictions
    Understand the trade-off between false positives (unnecessary further testing) and false negatives (missed conditions)
    Compare different model architectures more comprehensively
    Make informed decisions about model deployment in clinical settings

    AUC (Area Under the Curve) ROC (Receiver Operating Characteristic)

    The ROC curve is a graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) across different classification thresholds.
    AUC-ROC tells us:

    Classification Quality

    AUC ranges from 0 to 1 (0% to 100%)
    0.5 = random chance (like flipping a coin)
    1.0 = perfect classification
    < 0.5 = worse than random chance
    Threshold Independence

    Unlike accuracy, precision, or F1-score, AUC evaluates the model's performance across ALL possible classification thresholds
    This is especially important in medical imaging where you might want to adjust the threshold based on the clinical context
    For example, you might want:
    Higher sensitivity (recall) for screening tests
    Higher specificity (fewer false positives) for confirmatory tests
    Interpretation in Chest X-Ray Context An AUC of 0.85 would mean:

    If you randomly select one abnormal X-ray and one normal X-ray
    The model has an 85% chance of giving the abnormal X-ray a higher probability score than the normal one
    Looking at the code in trainer.py, we see two types of ROC AUC being calculated:

    # The model outputs probabilities and they're used to calculate ROC AUC
    roc_auc = roc_auc_score(targets_list, preds_list)  # Using raw probabilities, not thresholded predictions
    Advantages of using AUC in this chest X-ray project:

    Class Imbalance Handling

    Medical datasets often have imbalanced classes (more normal cases than abnormal)
    AUC remains effective even with imbalanced datasets
    This is why the code uses raw probabilities (preds_list) instead of thresholded predictions
    Clinical Relevance

    Helps clinicians understand trade-offs between:
    Sensitivity (catching all potential abnormalities)
    Specificity (minimizing false alarms)
    Allows for threshold adjustment based on clinical needs
    Model Comparison

    The code logs AUC to MLflow for both training and validation:
    mlflow.log_metrics(
        {
            "train_auc": train_auc,
            "val_auc": val_auc,
        },
        step=epoch,
    )
    This makes it easier to compare different model architectures or training runs
    Performance Monitoring

    The early stopping mechanism can use validation AUC to determine when to stop training
    Helps prevent overfitting while maintaining clinically relevant performance
    When interpreting the AUC values in this chest X-ray classification:

    AUC > 0.90: Excellent discrimination
    AUC 0.80-0.90: Good discrimination
    AUC 0.70-0.80: Fair discrimination
    AUC < 0.70: Poor discrimination
    Looking at the evaluation code, the system calculates both macro and weighted AUC:

    Macro AUC: Simple average across classes
    Weighted AUC: Average weighted by class frequency
    This comprehensive evaluation helps ensure the model is reliable enough for potential clinical applications while
    providing flexibility in setting operating points based on specific clinical needs.

    """

    def log_model_summary(self):
        # Use torchinfo instead of torchsummary as its newer and supports mac
        # GPUS
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

    def _calculate_metrics(
        self, targets_list: list, preds_list: list
    ) -> dict[str, float | list]:
        # Convert lists to numpy arrays first
        preds = np.array(preds_list)
        targets = np.array(targets_list)

        # Convert predictions to binary (0/1) using 0.5 threshold
        preds_binary = (preds > 0.5).astype(int)

        # Calculate per-class metrics
        n_classes = targets.shape[1]  # 14 categories
        per_class_metrics = {}

        for i in range(n_classes):
            class_metrics = {
                f"class_{i}_accuracy": accuracy_score(
                    targets[:, i], preds_binary[:, i]
                ),
                f"class_{i}_precision": precision_score(
                    targets[:, i], preds_binary[:, i], zero_division=0
                ),
                f"class_{i}_recall": recall_score(
                    targets[:, i], preds_binary[:, i], zero_division=0
                ),
                f"class_{i}_f1": f1_score(
                    targets[:, i], preds_binary[:, i], zero_division=0
                ),
                f"class_{i}_auc": roc_auc_score(targets[:, i], preds[:, i]),
            }
            per_class_metrics.update(class_metrics)

        # Calculate overall metrics
        # For multi-label classification, we use different averaging strategies
        overall_metrics = {
            "accuracy_subset": accuracy_score(
                targets, preds_binary
            ),  # Exact match accuracy
            "hamming_loss": 1
            - accuracy_score(
                targets.ravel(), preds_binary.ravel()
            ),  # Per-label accuracy
            "precision_micro": precision_score(
                targets, preds_binary, average="micro", zero_division=0
            ),
            "precision_macro": precision_score(
                targets, preds_binary, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                targets, preds_binary, average="micro", zero_division=0
            ),
            "recall_macro": recall_score(
                targets, preds_binary, average="macro", zero_division=0
            ),
            "f1_micro": f1_score(
                targets, preds_binary, average="micro", zero_division=0
            ),
            "f1_macro": f1_score(
                targets, preds_binary, average="macro", zero_division=0
            ),
            "roc_auc_micro": roc_auc_score(targets, preds, average="micro"),
            "roc_auc_macro": roc_auc_score(targets, preds, average="macro"),
        }

        # Combine all metrics
        metrics = {**overall_metrics, **per_class_metrics}

        # Store confusion matrices separately to avoid cluttering metrics
        confusion_matrices = []
        for i in range(n_classes):
            cm = confusion_matrix(
                targets[:, i], preds_binary[:, i], labels=[0, 1]
            ).tolist()
            confusion_matrices.append(cm)

        metrics["confusion_matrices"] = confusion_matrices

        return metrics

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

    def train_model(self, experiment_name: str = "ChestXRay"):

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

        with mlflow.start_run(log_system_metrics=True) as run:
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
                #
                # train_loss and val_loss - Critical for monitoring model convergence
                # train_accuracy and val_accuracy - Basic performance metric
                # val_auc (ROC-AUC) - Particularly important for medical imaging tasks as it measures
                # the model's ability to discriminate between classes regardless of threshold
                # metrics_to_log = {
                #     "train_loss": train_loss,
                #     "val_loss": val_loss,
                # }
                #
                # # Log all metrics except confusion matrices
                # for metric_name, value in train_metrics.items():
                #     if metric_name != "confusion_matrices":
                #         metrics_to_log[f"train_{metric_name}"] = value
                #
                # for metric_name, value in val_metrics.items():
                #     if metric_name != "confusion_matrices":
                #         metrics_to_log[f"val_{metric_name}"] = value

                # Log only key metrics matching logger.info output
                mlflow.log_metrics(
                    {
                        "train_accuracy_subset": train_metrics["accuracy_subset"],
                        "train_f1_micro": train_metrics["f1_micro"],
                        "val_accuracy_subset": val_metrics["accuracy_subset"],
                        "val_f1_micro": val_metrics["f1_micro"],
                        "val_roc_auc_micro": val_metrics["roc_auc_micro"],
                    },
                    step=epoch,
                )

                # Log a more concise message but with key metrics
                logger.info(
                    "Training epoch completed",
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs,
                    train_loss=round(train_loss, 4),
                    train_accuracy_subset=round(train_metrics["accuracy_subset"], 4),
                    train_f1_micro=round(train_metrics["f1_micro"], 4),
                    val_loss=round(val_loss, 4),
                    val_accuracy_subset=round(val_metrics["accuracy_subset"], 4),
                    val_f1_micro=round(val_metrics["f1_micro"], 4),
                    val_auc_micro=round(val_metrics["roc_auc_micro"], 4),
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

            return run.info.run_id
