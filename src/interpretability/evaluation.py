from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from medmnist import INFO


class MetricsReporter:
    def __init__(self) -> None:
        chest_info = INFO["chestmnist"]
        no_of_categories = len(chest_info["label"])
        class_names = [chest_info["label"][str(i)] for i in range(no_of_categories)]
        self.metrics: Dict[str, Any] = {}
        self.class_names = (
            class_names
            if class_names is not None
            else [str(i) for i in range(no_of_categories)]
        )

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
            # Ensure probabilities sum to 1 using softmax
            y_pred_proba_normalized = softmax(y_pred_proba, axis=1)

            y_pred: npt.NDArray[np.int_] = np.argmax(y_pred_proba, axis=1)
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)

            self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            self.metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True
            )

            # Calculate macro and weighted ROC AUC using normalized probabilities
            self.metrics["macro_roc_auc"] = roc_auc_score(
                y_true, y_pred_proba_normalized, multi_class="ovr", average="macro"
            )
            self.metrics["weighted_roc_auc"] = roc_auc_score(
                y_true, y_pred_proba_normalized, multi_class="ovr", average="weighted"
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
            xticklabels=self.class_names[: cm.shape[1]],
            yticklabels=self.class_names[: cm.shape[0]],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return plt.gcf()

    def log_to_mlflow(self):
        """Log metrics to MLflow for both binary and multi-class cases."""
        is_multiclass = "macro_roc_auc" in self.metrics

        summary_metrics = {}

        if is_multiclass:
            summary_metrics.update(
                {
                    "macro_roc_auc": self.metrics["macro_roc_auc"],
                    "weighted_roc_auc": self.metrics["weighted_roc_auc"],
                }
            )
        else:
            summary_metrics["roc_auc"] = self.metrics["roc_auc"]

        # Add accuracy to summary metrics
        report = self.metrics["classification_report"]
        summary_metrics["accuracy"] = report["accuracy"]

        # Log only summary metrics as MLflow metrics
        mlflow.log_metrics(summary_metrics)

        # Save detailed metrics as JSON artifact
        detailed_metrics = {
            "classification_report": self.metrics["classification_report"],
            "averages": {
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
            },
        }

        # Save detailed metrics as a JSON artifact
        mlflow.log_dict(detailed_metrics, "detailed_metrics.json")

        # Create consolidated per-class metrics visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

        metrics_data = []

        # Collect metrics for each class
        for i, class_label in enumerate(report.keys()):
            if class_label in ["accuracy", "macro avg", "weighted avg"]:
                continue

            class_metrics = report[class_label]
            class_name = (
                self.class_names[int(class_label)]
                if i < len(self.class_names)
                else class_label
            )

            metrics_data.append(
                {
                    "class_name": class_name,
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1_score": class_metrics["f1-score"],
                    "support": class_metrics["support"],
                }
            )

        # Sort by support count (descending)
        metrics_data.sort(key=lambda x: x["support"], reverse=True)

        # Unpack sorted data
        class_names = [d["class_name"] for d in metrics_data]
        precision_scores = [d["precision"] for d in metrics_data]
        recall_scores = [d["recall"] for d in metrics_data]
        f1_scores = [d["f1_score"] for d in metrics_data]
        support_values = [d["support"] for d in metrics_data]

        # Plot precision, recall, and f1 scores
        x = np.arange(len(class_names))
        width = 0.25

        ax1.bar(x - width, precision_scores, width, label="Precision")
        ax1.bar(x, recall_scores, width, label="Recall")
        ax1.bar(x + width, f1_scores, width, label="F1-Score")
        ax1.set_ylabel("Score")
        ax1.set_title("Precision, Recall, and F1-Score by Class (Sorted by Support)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot support values
        ax2.bar(x, support_values)
        ax2.set_ylabel("Support")
        ax2.set_title("Class Distribution (Support)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        mlflow.log_figure(fig, "class_metrics.png")
        plt.close()

        # Log confusion matrix plot
        cm_fig = self.plot_confusion_matrix()
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close()


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
