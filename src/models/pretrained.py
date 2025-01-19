import pytorch_lightning as pl
import structlog
import torch.nn as nn
import torchvision.models as models


logger = structlog.get_logger()


class ChestNetResnet(pl.LightningModule):
    def __init__(self, num_classes=14, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        # Modify the first conv to accept 1 channel
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final layer
        self.backbone.fc = nn.Linear(512, num_classes)
