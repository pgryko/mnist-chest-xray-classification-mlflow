from scipy.special import expit as sigmoid

import mlflow
import structlog
import torch

from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule_lightning import ChestDataModuleLightning
from src.interpretability.evaluation import evaluate_model, MetricsReporter

logger = structlog.get_logger()


def main(run_id: str):
    # Instantiate configs
    train_config = TrainingConfig(num_epochs=100)
    path_config = PathConfig()

    # Create data module
    data_module = ChestDataModuleLightning(train_config, path_config)
    data_module.setup(stage="test")

    test_loader = data_module.test_dataloader()

    if test_loader is None or not hasattr(test_loader.dataset, "__len__"):
        raise ValueError("Test dataset not properly initialized")

    # Prepare device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    mlflow.set_tracking_uri(path_config.mlflow_tracking_uri)

    mlflow.set_experiment("ChestXRayPytorchLightning")

    #     loaded_model = load_best_model(
    #         experiment_name="ChestXRay",
    #         metric="val_loss",
    #         tracking_uri=path_config.mlflow_tracking_uri,
    #     )

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Evaluate on the test set in a separate run
    with mlflow.start_run(
        # run_id="55661eb3fed343cea2204b1f34865414",
    ):
        y_true, y_prob = evaluate_model(model, test_loader, device)

        # Convert logits to probability
        y_pred_proba = sigmoid(y_prob)

        reporter = MetricsReporter()
        reporter.calculate_metrics(y_true, y_pred_proba)
        reporter.log_to_mlflow()


if __name__ == "__main__":
    main(run_id="1be777d365ee4a1a90eac621e75ea49e")
