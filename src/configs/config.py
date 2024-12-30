from dataclasses import dataclass, field
from typing import List
from decouple import config 

@dataclass
class TrainingConfig:
    # Basic training configs
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 7  # for early stopping

    # Data augmentation settings
    rotate_limit: int = 15
    brightness: float = 0.2
    contrast: float = 0.2

    # List of possible learning rates or batch sizes
    lr_candidates: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    batch_candidates: List[int] = field(default_factory=lambda: [32, 64, 128])


@dataclass
class PathConfig:
    # Paths to data, logs, etc.
    dataset_root: str = config('DATASET_ROOT', default='./data')
    checkpoint_dir: str = config('CHECKPOINT_DIR', default='./checkpoints')
    mlflow_tracking_uri: str = config('MLFLOW_TRACKING_URI', default='file:./mlruns')



# Example: how to instantiate them in code
# config = TrainingConfig()
# paths = PathConfig()
