from scipy.special import expit as sigmoid
import mlflow
import structlog
import torch
from pathlib import Path
from typing import Optional

from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule_lightning import ChestDataModuleLightning
from src.interpretability.evaluation import evaluate_model, MetricsReporter
from src.interpretability.explainers import XAIExplainer
from src.utils.visualization import plot_multiple_explanations

logger = structlog.get_logger()


def generate_explanations(model, data_loader, device, save_dir, num_samples=5):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    xai_explainer = XAIExplainer(model, device)

    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    for idx in range(num_samples):
        image = images[idx : idx + 1]
        label = labels[idx]

        # Get explanations
        gradcam_exp = xai_explainer.compute_gradcam(image, target_class=0)
        ig_exp = xai_explainer.compute_integrated_gradients(image, target_class=0)

        # Create explanation dictionary
        explanations = {
            "GradCAM": gradcam_exp,
            "Integrated Gradients": ig_exp.squeeze().abs().cpu().numpy(),
        }

        # Plot and save
        plot_multiple_explanations(
            image.squeeze().cpu().numpy(),
            explanations,
            save_path=save_dir / f"sample_{idx}_explanations.png",
            show=False,
        )


def main(run_id: str, explanation_dir: Optional[str] = None):
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

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Evaluate on the test set in a separate run
    with mlflow.start_run():
        # Generate standard metrics
        y_true, y_prob = evaluate_model(model, test_loader, device)
        y_pred_proba = sigmoid(y_prob)

        reporter = MetricsReporter()
        reporter.calculate_metrics(y_true, y_pred_proba)
        reporter.log_to_mlflow()

        # Generate explanations if directory is provided
        if explanation_dir:
            explanation_path = Path(explanation_dir)
            generate_explanations(
                model, test_loader, device, explanation_path, num_samples=5
            )

            # Log explanations to MLflow
            mlflow.log_artifacts(explanation_dir, artifact_path="explanations")

            logger.info(
                "Generated explanations",
                explanation_dir=str(explanation_path),
                num_samples=5,
            )


if __name__ == "__main__":
    main(run_id="e53f7b541ec3471382fc3e12e77ef547", explanation_dir="explanations")
