from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from medmnist import INFO


class MetricsReporter:
    """Reports and visualizes metrics for multi-label chest X-ray classification."""

    def __init__(self) -> None:
        """Initialize reporter with chest X-ray specific class names."""
        chest_info = INFO["chestmnist"]
        self.disease_names = [
            chest_info["label"][str(i)] for i in range(len(chest_info["label"]))
        ]
        self.metrics: Dict[str, Any] = {}

    def calculate_metrics(
        self,
        y_true: npt.NDArray[np.int_],
        y_pred_proba: npt.NDArray[np.float64],
        threshold: float = 0.5,
    ) -> None:
        """Calculate comprehensive metrics for multi-label classification.

        Args:
            y_true: Ground truth labels (shape: n_samples x n_classes)
            y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
            threshold: Classification threshold for converting probabilities to binary predictions
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        self.y_true = y_true  # Store for later use
        self.y_pred = (y_pred_proba >= threshold).astype(int)  # Store for later use

        # Calculate metrics for each condition
        self.metrics["per_condition"] = {}
        self.metrics["confusion_matrices"] = (
            {}
        )  # New dictionary for per-class confusion matrices

        # Add normal class metrics
        normal_cases = np.all(y_true == 0, axis=1)
        predicted_normal = np.all(y_pred == 0, axis=1)
        normal_proba = 1 - np.max(y_pred_proba, axis=1)  # Probability of being normal

        # Add normal to per_condition metrics
        self.metrics["per_condition"]["Normal"] = {
            "auc_roc": roc_auc_score(normal_cases, normal_proba),
            "avg_precision": average_precision_score(normal_cases, normal_proba),
            "support": int(np.sum(normal_cases)),
            "precision": precision_recall_curve(normal_cases, normal_proba)[0],
            "recall": precision_recall_curve(normal_cases, normal_proba)[1],
        }

        # Add normal to confusion matrices
        self.metrics["confusion_matrices"]["Normal"] = confusion_matrix(
            normal_cases, predicted_normal
        )

        # Add metrics for each condition including normal
        for idx, condition in enumerate(self.disease_names):
            # For disease conditions
            condition_true = y_true[:, idx]
            condition_pred = y_pred[:, idx]
            condition_proba = y_pred_proba[:, idx]

            # Calculate confusion matrix values
            tn, fp, fn, tp = confusion_matrix(condition_true, condition_pred).ravel()

            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = (
                2 * (precision * sensitivity) / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0
            )

            prec_curve, recall_curve, _ = precision_recall_curve(
                condition_true, condition_proba
            )

            self.metrics["per_condition"][condition] = {
                "auc_roc": roc_auc_score(condition_true, condition_proba),
                "avg_precision": average_precision_score(
                    condition_true, condition_proba
                ),
                "support": int(np.sum(condition_true)),
                "precision": precision,
                "recall": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "f1_score": f1,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "precision_curve": prec_curve,
                "recall_curve": recall_curve,
            }

            self.metrics["confusion_matrices"][condition] = confusion_matrix(
                condition_true, condition_pred
            )

        # Calculate overall metrics
        self.metrics["overall"] = {
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.disease_names, output_dict=True
            ),
            "macro_auc_roc": roc_auc_score(y_true, y_pred_proba, average="macro"),
            "micro_auc_roc": roc_auc_score(y_true, y_pred_proba, average="micro"),
            "weighted_auc_roc": roc_auc_score(y_true, y_pred_proba, average="weighted"),
            "samples_with_conditions": int(np.sum(np.any(y_true == 1, axis=1))),
            "normal_samples": int(np.sum(np.all(y_true == 0, axis=1))),
            "total_samples": len(y_true),
        }

        # Calculate normal vs any disease metrics
        normal_cases = np.all(y_true == 0, axis=1)
        predicted_normal = np.all(y_pred == 0, axis=1)

        self.metrics["normal_detection"] = {
            "confusion_matrix": confusion_matrix(normal_cases, predicted_normal),
            "classification_report": classification_report(
                normal_cases,
                predicted_normal,
                target_names=["Any Disease", "Normal"],
                output_dict=True,
            ),
        }

    def plot_confusion_matrix(self):
        """Plot a single consolidated confusion matrix for all classes.

        Returns:
            matplotlib.figure.Figure: The confusion matrix plot
        """
        # Create a consolidated confusion matrix
        y_true_flat = np.argmax(self.y_true, axis=1)
        y_pred_flat = np.argmax(self.y_pred, axis=1)

        cm = confusion_matrix(y_true_flat, y_pred_flat)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.disease_names,
            yticklabels=self.disease_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return plt.gcf()

    def plot_metrics(self) -> Dict[str, plt.Figure]:
        """Create visualizations for all metrics.

        Returns:
            Dictionary containing all generated figures
        """
        figures = {}

        # 1. Plot condition-wise metrics
        fig_metrics, ax = plt.subplots(figsize=(12, 6))
        conditions = list(self.metrics["per_condition"].keys())
        auc_scores = [self.metrics["per_condition"][c]["auc_roc"] for c in conditions]

        sns.barplot(x=conditions, y=auc_scores)
        plt.xticks(rotation=45, ha="right")
        plt.title("AUC-ROC Scores by Condition")
        plt.tight_layout()
        figures["auc_by_condition"] = fig_metrics

        # 2. Plot normal vs disease confusion matrix
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            self.metrics["normal_detection"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Disease", "Normal"],
            yticklabels=["Disease", "Normal"],
        )
        plt.title("Normal vs Disease Detection")
        plt.tight_layout()
        figures["normal_detection"] = fig_cm

        # 3. Plot support distribution
        fig_support, ax = plt.subplots(figsize=(12, 6))
        supports = [self.metrics["per_condition"][c]["support"] for c in conditions]
        sns.barplot(x=conditions, y=supports)
        plt.xticks(rotation=45, ha="right")
        plt.title("Number of Cases by Condition")
        plt.tight_layout()
        figures["condition_support"] = fig_support

        # Create consolidated per-class metrics visualizations
        # Create consolidated per-class metrics visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

        metrics_data = []
        report = self.metrics["overall"]["classification_report"]

        # Collect metrics for each disease class
        for class_label in self.disease_names:
            if class_label in report:  # Ensure the class exists in the report
                class_metrics = report[class_label]
                metrics_data.append(
                    {
                        "class_name": class_label,
                        "precision": class_metrics["precision"],
                        "recall": class_metrics["recall"],
                        "f1_score": class_metrics["f1-score"],
                        "support": class_metrics["support"],
                    }
                )

        if not metrics_data:
            plt.close(fig)
            return figures

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

        # Add color scheme for better visibility
        ax1.bar(x - width, precision_scores, width, label="Precision", color="#2ecc71")
        ax1.bar(x, recall_scores, width, label="Recall", color="#3498db")
        ax1.bar(x + width, f1_scores, width, label="F1-Score", color="#e74c3c")

        ax1.set_ylabel("Score")
        ax1.set_title("Precision, Recall, and F1-Score by Class", fontsize=12, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)  # Set y-axis limits for scores

        # Plot support values
        support_bars = ax2.bar(x, support_values, color="#9b59b6")
        ax2.set_ylabel("Number of Samples")
        ax2.set_title("Class Distribution (Support)", fontsize=12, pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Add value labels on support bars
        for bar in support_bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        figures["class_metrics"] = fig

        # Log confusion matrices
        confusion_matrix_fig = self.plot_confusion_matrix()
        figures["confusion_matrix"] = confusion_matrix_fig

        return figures

    def log_to_mlflow(self) -> None:
        """Log all metrics and visualizations to MLflow."""
        # Log overall metrics
        # mlflow.log_metrics(
        #     {
        #         "macro_auc_roc": self.metrics["overall"]["macro_auc_roc"],
        #         "micro_auc_roc": self.metrics["overall"]["micro_auc_roc"],
        #         "weighted_auc_roc": self.metrics["overall"]["weighted_auc_roc"],
        #         "normal_detection_accuracy": self.metrics["normal_detection"][
        #             "classification_report"
        #         ]["accuracy"],
        #     }
        # )

        # Log per-condition metrics
        # for condition, metrics in self.metrics["per_condition"].items():
        #     mlflow.log_metrics(
        #         {
        #             f"auc_roc_{condition.lower()}": metrics["auc_roc"],
        #             f"avg_precision_{condition.lower()}": metrics["avg_precision"],
        #         }
        #     )

        # Log detailed metrics as JSON
        mlflow.log_dict(self.metrics["overall"], "overall_metrics.json")
        mlflow.log_dict(
            self.metrics["normal_detection"], "normal_detection_metrics.json"
        )

        # Log visualizations
        figures = self.plot_metrics()
        for name, fig in figures.items():
            if isinstance(fig, plt.Figure):  # Add type check
                mlflow.log_figure(fig, f"{name}.png")
                plt.close(fig)

    def print_summary(self) -> None:
        """Print a human-readable summary of the metrics."""
        print("\n=== Model Performance Summary ===\n")

        print("Overall Metrics:")
        print(f"Macro AUC-ROC: {self.metrics['overall']['macro_auc_roc']:.3f}")
        print(f"Micro AUC-ROC: {self.metrics['overall']['micro_auc_roc']:.3f}")
        print(f"Total samples: {self.metrics['overall']['total_samples']}")
        print(f"Normal samples: {self.metrics['overall']['normal_samples']}")
        print(
            f"Samples with conditions: {self.metrics['overall']['samples_with_conditions']}"
        )

        print("\nPer-Condition Performance:")
        for condition, metrics in self.metrics["per_condition"].items():
            print(f"\n{condition}:")
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  Avg Precision: {metrics['avg_precision']:.3f}")
            print(f"  Support: {metrics['support']}")

        print("\nNormal vs Disease Detection:")
        report = self.metrics["normal_detection"]["classification_report"]
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Normal Precision: {report['Normal']['precision']:.3f}")
        print(f"Normal Recall: {report['Normal']['recall']:.3f}")


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
