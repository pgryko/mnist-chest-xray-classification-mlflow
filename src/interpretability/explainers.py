import shap
import torch
from captum.attr import LayerGradCam


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
