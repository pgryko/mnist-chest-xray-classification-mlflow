import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Union
import cv2

from src.models.pretrained import ChestNetResnet


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)
    return image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image (H, W) or (H, W, C)
        heatmap: Heatmap to overlay (H, W)
        alpha: Transparency of the overlay
        colormap: OpenCV colormap to use

    Returns:
        Overlaid image
    """
    # Ensure image is in uint8 format and correct shape
    image = normalize_image(image)

    # Convert grayscale to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)

    # Ensure heatmap is the same size as the image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap to 0-255 range
    heatmap = normalize_image(heatmap)

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Ensure both images are in the same format
    image = image.astype(np.float32)
    heatmap_colored = heatmap_colored.astype(np.float32)

    # Overlay images
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlaid.astype(np.uint8)


def plot_explanation(
    image: Union[np.ndarray, torch.Tensor],
    explanation: Union[np.ndarray, torch.Tensor],
    method: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot an explanation alongside the original image.

    Args:
        image: Original image
        explanation: Explanation to plot
        method: Name of the explanation method
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy()
    if isinstance(explanation, torch.Tensor):
        explanation = explanation.squeeze().cpu().numpy()

    # Ensure image is 2D
    if len(image.shape) > 2:
        image = image.squeeze()
    if len(explanation.shape) > 2:
        explanation = explanation.squeeze()

    plt.figure(figsize=(12, 4))

    # Plot original image
    plt.subplot(131)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Plot explanation
    plt.subplot(132)
    plt.imshow(explanation, cmap="jet")
    plt.title(f"{method} Explanation")
    plt.axis("off")

    # Plot overlay
    plt.subplot(133)
    overlaid = overlay_heatmap(image, explanation)
    plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    plt.title("Overlaid Explanation")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_explanations(
    image: np.ndarray,
    explanations: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot multiple explanations for the same image.

    Args:
        image: Original image
        explanations: Dictionary of explanation name to explanation array
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    n_explanations = len(explanations)
    plt.figure(figsize=(4 * (n_explanations + 1), 4))

    # Plot original image
    plt.subplot(1, n_explanations + 1, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Plot each explanation
    for idx, (name, explanation) in enumerate(explanations.items(), start=2):
        plt.subplot(1, n_explanations + 1, idx)
        overlaid = overlay_heatmap(image, explanation)
        plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def explain_chest_xray(model: ChestNetResnet, image: torch.Tensor, target_class: int):
    """
    Generate and visualize explanations for a chest X-ray prediction.

    Args:
        model: Trained ChestNetResnet model
        image: Input image tensor
        target_class: Target class to explain
    """
    # Get GradCAM explanation
    gradcam_exp = model.explain_prediction(image, target_class, method="gradcam")
    plot_explanation(
        image.squeeze().cpu().numpy(),
        gradcam_exp,
        "Grad-CAM",
        save_path="gradcam_explanation.png",
    )

    # Get Integrated Gradients explanation
    ig_exp = model.explain_prediction(
        image, target_class, method="integrated_gradients"
    )
    plot_explanation(
        image.squeeze().cpu().numpy(),
        ig_exp.squeeze().abs().cpu().numpy(),
        "Integrated Gradients",
        save_path="ig_explanation.png",
    )

    # Get activation maps
    activation_maps = model.get_activation_maps(image)

    # Plot selected activation maps
    plt.figure(figsize=(15, 5))
    for i, (name, activation) in enumerate(list(activation_maps.items())[:3]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(activation[0, 0].cpu().numpy(), cmap="viridis")
        plt.title(f"Activation: {name}")
        plt.axis("off")
    plt.savefig("activation_maps.png")
    plt.close()
