import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
import torch
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


class MetricsReporter:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)

        self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        self.metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True
        )
        self.metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        self.metrics["precision"] = precision
        self.metrics["recall"] = recall

    def plot_confusion_matrix(self):
        cm = self.metrics["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        return plt.gcf()

    def log_to_mlflow(self):
        # Summaries
        mlflow.log_metric("roc_auc", self.metrics["roc_auc"])

        # You can also log the confusion matrix plot
        cm_fig = self.plot_confusion_matrix()
        mlflow.log_figure(cm_fig, "confusion_matrix.png")


def evaluate_model(model, data_loader, device):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            y_true.extend(target.numpy().astype(int))
            y_prob.extend(output.cpu().numpy())
    y_prob = np.array(y_prob).ravel()
    return np.array(y_true), y_prob
