from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerGradCam
import shap
import cv2


class ModelInterpreter:
    def __init__(self, model, target_layer, device):
        self.model = model.to(device)
        self.device = device
        self.grad_cam = LayerGradCam(self.model, target_layer)

    def generate_grad_cam(self, input_tensor):
        # input_tensor: shape [N, C, H, W]
        attribution = self.grad_cam.attribute(input_tensor.to(self.device))
        # Normalize or visualize as needed
        return attribution


class SHAPInterpreter:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.e = shap.DeepExplainer(
            self.model, torch.zeros((1, 1, 64, 64), dtype=torch.float).to(self.device)
        )

    def explain(self, input_tensor):
        # input_tensor: shape [N, C, H, W]
        shap_values = self.e.shap_values(input_tensor.to(self.device))
        return shap_values


class XAIExplainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.integrated_gradients = IntegratedGradients(model)
        # Get the last convolutional layer for Grad-CAM
        self.last_conv_layer = self._get_last_conv_layer()
        self.grad_cam = LayerGradCam(model, self.last_conv_layer)

    def _get_last_conv_layer(self) -> nn.Module:
        """Returns the last convolutional layer of the model."""
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        return conv_layers[-1]

    def compute_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM activation maps for the specified class.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            resize_to: Optional tuple of (height, width) to resize the output

        Returns:
            Numpy array containing the heatmap
        """
        self.model.eval()
        attribution = self.grad_cam.attribute(input_tensor, target=target_class)

        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap / (np.max(heatmap) + 1e-10)  # Normalize

        if resize_to:
            heatmap = cv2.resize(heatmap, resize_to)

        return heatmap

    def compute_integrated_gradients(
        self, input_tensor: torch.Tensor, target_class: int, n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution for the specified class.

        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            n_steps: Number of steps for path integral

        Returns:
            Attribution tensor
        """
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        attributions = self.integrated_gradients.attribute(
            input_tensor, baseline, target=target_class, n_steps=n_steps
        )
        return attributions

    def compute_deep_shap(
        self,
        input_tensor: torch.Tensor,
        background_data: torch.Tensor,
        n_samples: int = 50,
    ) -> np.ndarray:
        """
        Compute Deep SHAP values for the input.

        Args:
            input_tensor: Input image tensor
            background_data: Background data for SHAP
            n_samples: Number of background samples to use

        Returns:
            SHAP values as numpy array
        """
        self.model.eval()
        explainer = shap.DeepExplainer(self.model, background_data[:n_samples])
        shap_values = explainer.shap_values(input_tensor)
        return shap_values
