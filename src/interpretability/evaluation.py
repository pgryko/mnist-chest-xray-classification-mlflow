import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import mlflow


from typing import Dict, Any
import numpy.typing as npt


class MetricsReporter:
    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}

    def calculate_metrics(
        self, y_true: npt.NDArray[np.int_], y_pred_proba: npt.NDArray[np.float64]
    ) -> None:
        """Calculate metrics for both binary and multi-class classification.

        Args:
            y_true: Ground truth labels
            y_pred_proba: Model predictions (probabilities)
        """
        # Check if this is binary or multi-class classification
        is_multiclass: bool = len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1

        if is_multiclass:
            # Multi-class case
            y_pred: npt.NDArray[np.int_] = np.argmax(y_pred_proba, axis=1)
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)

            self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            self.metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True
            )

            # Calculate macro and weighted ROC AUC
            self.metrics["macro_roc_auc"] = roc_auc_score(
                y_true, y_pred_proba, multi_class="ovr", average="macro"
            )
            self.metrics["weighted_roc_auc"] = roc_auc_score(
                y_true, y_pred_proba, multi_class="ovr", average="weighted"
            )
        else:
            # Binary case
            y_pred: npt.NDArray[np.int_] = (y_pred_proba >= 0.5).astype(int)

            self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            self.metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True
            )
            self.metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    def plot_confusion_matrix(self):
        """Plot confusion matrix using seaborn.

        Returns:
            matplotlib.figure.Figure: The confusion matrix plot
        """
        cm = self.metrics["confusion_matrix"]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(cm.shape[1]),
            yticklabels=range(cm.shape[0]),
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        return plt.gcf()

    def log_to_mlflow(self):
        """Log metrics to MLflow for both binary and multi-class cases."""
        is_multiclass = "macro_roc_auc" in self.metrics

        if is_multiclass:
            # Log multi-class metrics
            mlflow.log_metric("macro_roc_auc", self.metrics["macro_roc_auc"])
            mlflow.log_metric("weighted_roc_auc", self.metrics["weighted_roc_auc"])
        else:
            # Log binary metrics
            mlflow.log_metric("roc_auc", self.metrics["roc_auc"])

        # Log classification report metrics
        report = self.metrics["classification_report"]
        mlflow.log_metric("accuracy", report["accuracy"])

        # Log per-class metrics
        for class_label in report.keys():
            if class_label in ["accuracy", "macro avg", "weighted avg"]:
                continue
            class_metrics = report[class_label]
            mlflow.log_metric(
                f"precision_class_{class_label}", class_metrics["precision"]
            )
            mlflow.log_metric(f"recall_class_{class_label}", class_metrics["recall"])
            mlflow.log_metric(
                f"f1_score_class_{class_label}", class_metrics["f1-score"]
            )
            mlflow.log_metric(f"support_class_{class_label}", class_metrics["support"])

        # Log macro and weighted averages
        for avg_type in ["macro avg", "weighted avg"]:
            avg_metrics = report[avg_type]
            prefix = avg_type.replace(" ", "_")
            mlflow.log_metric(f"{prefix}_precision", avg_metrics["precision"])
            mlflow.log_metric(f"{prefix}_recall", avg_metrics["recall"])
            mlflow.log_metric(f"{prefix}_f1_score", avg_metrics["f1-score"])

        # Log confusion matrix plot
        cm_fig = self.plot_confusion_matrix()
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close()  # Close the figure to free memory


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Evaluate a PyTorch model and return true labels and predicted probabilities.

    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run the model on (CPU or GPU)

    Returns:
        A tuple of (true labels, predicted probabilities) as numpy arrays
    """
    model.eval()
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)

            probs = output.cpu().numpy()

            # Keep multi-class information
            y_true.extend(target.numpy())
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_prob)
