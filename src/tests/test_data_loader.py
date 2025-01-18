import pytest
import torch

from src.configs.config import TrainingConfig, PathConfig
from src.data.datamodule_lightning import ChestDataModuleLightning


@pytest.mark.parametrize("stage", ["fit", "test"])
def test_chest_data_module_shapes(stage):
    # 1. Instantiate configs
    train_config = TrainingConfig(batch_size=32)  # or your preferred batch size
    path_config = PathConfig()

    # 2. Instantiate data module
    dm = ChestDataModuleLightning(train_config, path_config)
    dm.setup(stage)

    # 3. Depending on stage, pick the right dataloader
    if stage == "fit":
        dataloader = dm.train_dataloader()
    else:
        dataloader = dm.test_dataloader()

    # 4. Fetch one batch
    x, y = next(iter(dataloader))

    # 5. Assert expected shapes
    # For ChestMNIST multi-label: (batch_size, 1, 64, 64) for x; (batch_size, 14) for y.
    assert (
        x.dim() == 4
    ), f"Expected 4D input (batch, channels, height, width), got {x.shape}"
    assert x.size(1) == 1, f"Expected 1 channel for ChestMNIST, got {x.shape[1]}"
    assert (
        x.size(2) == 64 and x.size(3) == 64
    ), f"Expected 64x64 images, got {x.shape[2:]}"

    assert y.dim() == 2, f"Labels should be 2D, got {y.shape}"
    assert y.size(1) == 14, f"Expected 14-label multi-label output, got {y.shape[1]}"

    # (Optional) Test that data types are as expected
    assert x.dtype == torch.float32, f"Expected x.dtype to be float32, got {x.dtype}"
    # Labels often are Long or Float. Adjust as needed:
    assert y.dtype in [torch.float32, torch.int64], f"Unexpected label dtype: {y.dtype}"
