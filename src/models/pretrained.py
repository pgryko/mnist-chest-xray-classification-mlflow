import structlog
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.chestnets_lighning import ChestNetBase

logger = structlog.get_logger()


class ChestNetResnet(ChestNetBase):
    # Input Size Compatibility: The ResNet-18 model is compatible with 64x64 inputs due to the
    # adaptive average pooling layer, which adjusts to varying spatial dimensions.
    #
    # Pretrained Weights Handling: The first convolutional layer's weights are initialized by
    # averaging the pretrained RGB weights, preserving some pretrained features even with grayscale input.
    #
    # Logging and Metrics: The existing logging and metrics setup (via MLflow) remains unchanged,
    # leveraging the base class's functionality.

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_classes: int = 14,
        pos_weight: torch.Tensor = None,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_classes=num_classes,
            pos_weight=pos_weight,
        )
        self.model_name = "ResNet18"
        self.model_details = {
            "architecture": "ResNet-18",
            "pretrained": pretrained,
            "input_channels": 1,
            "modified_layers": ["conv1", "fc"],
        }

        # Load pretrained ResNet-18
        backbone = models.resnet18(pretrained=pretrained)

        # Modify first convolutional layer for grayscale input
        original_conv1 = backbone.conv1
        self.backbone = backbone
        self.backbone.conv1 = nn.Conv2d(
            1,  # Input channels changed to 1
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # Initialize weights from pretrained model
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Define feature extractor
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x
