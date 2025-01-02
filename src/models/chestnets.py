import torch
import torch.nn as nn
import torch.nn.functional as F


class ChestNetS(nn.Module):
    """A small Convolutional Neural Network for chest X-ray classification.

    This network implements a simple CNN architecture with three convolutional blocks,
    followed by a fully connected classifier. It is designed to be lightweight while
    maintaining reasonable performance on chest X-ray classification tasks.

    Architecture Details:
        - Input: Single channel (grayscale) images
        - Three Conv Blocks: Progressive filter expansion (32→64→128)
        - Each Block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        - Classifier: Two fully connected layers with dropout
        - Output: Single sigmoid unit for binary classification

    Attributes:
        model_name (str): Identifier for the model ("ChestNetS")
        model_details (dict): Configuration parameters and architecture details
        features (nn.Sequential): Convolutional layers for feature extraction
        classifier (nn.Sequential): Fully connected layers for classification

    Example:
        >>> model = ChestNetS()
        >>> batch_size, channels, height, width = 32, 1, 64, 64
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    """

    def __init__(self):
        super(ChestNetS, self).__init__()
        self.model_name = "ChestNetS"
        self.model_details = {
            "architecture": "Small CNN",
            "input_channels": 1,
            "conv_layers": 3,
            "initial_filters": 32,
            "max_filters": 128,
            "dropout_rate": 0.5,
            "final_activation": "sigmoid"
        }

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 32x32
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 16x16
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 8x8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x


class ChestNetM(nn.Module):
    """A medium-sized Convolutional Neural Network with residual connections for chest X-ray classification.

    This network implements a ResNet-style architecture with three residual blocks and adaptive average pooling.
    It is designed to process grayscale chest X-ray images and output binary classifications.

    Architecture Details:
        - Input: Single channel (grayscale) images
        - Initial Conv Layer: 64 filters with 3x3 kernel
        - Three Residual Blocks: Progressive filter expansion (64→128→256)
        - Pooling: Max pooling after each residual block
        - Final Layer: Adaptive average pooling followed by fully connected layer
        - Output: Single sigmoid unit for binary classification

    Attributes:
        model_name (str): Identifier for the model ("ChestNetM")
        model_details (dict): Comprehensive configuration parameters and architecture details
        initial (nn.Sequential): Initial convolutional block
        layer1 (nn.Sequential): First residual block with pooling
        layer2 (nn.Sequential): Second residual block with pooling
        layer3 (nn.Sequential): Third residual block with pooling
        classifier (nn.Sequential): Final classification layers

    Example:
        >>> model = ChestNetM()
        >>> batch_size, channels, height, width = 32, 1, 64, 64
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    """

    def __init__(self):
        super(ChestNetM, self).__init__()
        self.model_name = "ChestNetM"
        self.model_details = {
            "architecture": "Medium ResNet",
            "input_channels": 1,
            "initial_filters": 64,
            "max_filters": 256,
            "residual_blocks": 3,
            "dropout_rate": 0.5,
            "pooling": "adaptive_avg",
            "final_activation": "sigmoid",
            "description": "Medium-sized CNN with residual connections",
            "params": {
                "conv_layers": [64, 128, 256],
                "kernel_size": 3,
                "padding": 1,
                "pool_size": 2
            }
        }

        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64), nn.MaxPool2d(2, 2)  # Output: 32x32
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128), nn.MaxPool2d(2, 2)  # Output: 16x16
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256), nn.MaxPool2d(2, 2)  # Output: 8x8
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class ChestNetL(nn.Module):
    """A large Convolutional Neural Network with residual connections and attention mechanisms for chest X-ray classification.

    This network implements an advanced architecture combining ResNet-style residual blocks with
    channel attention mechanisms. It is designed to handle complex feature extraction from chest
    X-ray images through its deep structure and attention-based feature refinement.

    Architecture Details:
        - Input: Single channel (grayscale) images
        - Initial Conv Layer: 64 filters with 3x3 kernel
        - Three Residual Blocks: Progressive filter expansion (64→128→256)
        - Attention Mechanism: Channel attention after each residual block
        - Pooling: Max pooling after each block combination
        - Classifier: Two fully connected layers (256->128->1) with dropout
        - Output: Single sigmoid unit for binary classification

    Attributes:
        model_name (str): Identifier for the model ("ChestNetL")
        model_details (dict): Comprehensive configuration parameters and architecture details
        initial (nn.Sequential): Initial convolutional block
        layer1 (nn.Sequential): First residual block with attention and pooling
        layer2 (nn.Sequential): Second residual block with attention and pooling
        layer3 (nn.Sequential): Third residual block with attention and pooling
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling layer
        classifier (nn.Sequential): Final classification layers

    Example:
        >>> model = ChestNetL()
        >>> batch_size, channels, height, width = 32, 1, 64, 64
        >>> x = torch.randn(batch_size, channels, height, width)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    """
    def __init__(self):
        super(ChestNetL, self).__init__()
        self.model_name = "ChestNetL"
        self.model_details = {
            "architecture": "Large ResNet with Attention",
            "input_channels": 1,
            "initial_filters": 64,
            "max_filters": 256,
            "residual_blocks": 3,
            "attention_blocks": 3,
            "dropout_rate": 0.5,
            "pooling": "adaptive_avg",
            "final_activation": "sigmoid",
            "description": "Large CNN with residual connections and attention mechanisms",
            "params": {
                "conv_layers": [64, 128, 256],
                "kernel_size": 3,
                "padding": 1,
                "pool_size": 2,
                "hidden_units": [256, 128, 1],
                "attention_reduction": 8
            },
            "architecture_details": {
                "initial_conv": "64 filters, 3x3 kernel",
                "residual_blocks": "3 blocks with increasing filters (64→128→256)",
                "attention_mechanism": "Channel attention after each residual block",
                "classifier": "256->128->1 with dropout layers"
            }
        }

        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64), AttentionBlock(64), nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128), AttentionBlock(128), nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256), AttentionBlock(256), nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
                representing grayscale chest X-ray images.

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1), where each value
                represents the probability of the input belonging to the positive class.
        """
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
