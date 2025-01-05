import json
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.configs.config import PathConfig


class ModelCheckpointer:
    def __init__(self, base_dir="checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.version_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self):
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=4)

    def save_checkpoint(self, state, metric_value, is_best=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = len(self.versions) + 1

        checkpoint_info = {
            "timestamp": timestamp,
            "metric_value": float(metric_value),
            "is_best": is_best,
        }

        checkpoint_path = self.base_dir / f"v{version}_checkpoint_{timestamp}.pth"
        torch.save(state, checkpoint_path)

        self.versions[f"v{version}"] = checkpoint_info
        self._save_versions()

        if is_best:
            best_path = self.base_dir / "best_model.pth"
            torch.save(state, best_path)

        return checkpoint_path


def load_pytorch_model(run_id, tracking_uri=PathConfig.mlflow_tracking_uri):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Load the model
    loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
    return loaded_model


def load_best_model(experiment_name="ChestXRay", metric="val_loss", tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            f"metrics.{metric} ASC"
        ],  # ASC for metrics like loss, DESC for metrics like accuracy
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    # Get the best run (first run since we ordered by metric)
    best_run = runs[0]

    # Load the model from the best run
    loaded_model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/best_model")

    print(
        f"Loaded model from run {best_run.info.run_id} with {metric}: {best_run.data.metrics.get(metric)}"
    )
    return loaded_model
