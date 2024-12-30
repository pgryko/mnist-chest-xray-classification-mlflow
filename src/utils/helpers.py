import torch
import json
from pathlib import Path
from datetime import datetime


class ModelCheckpointer:
    def __init__(self, base_dir="checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.version_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self):
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=4)

    def save_checkpoint(self, state, metric_value, is_best=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = len(self.versions) + 1

        checkpoint_info = {
            "timestamp": timestamp,
            "metric_value": float(metric_value),
            "is_best": is_best,
        }

        checkpoint_path = self.base_dir / f"v{version}_checkpoint_{timestamp}.pth"
        torch.save(state, checkpoint_path)

        self.versions[f"v{version}"] = checkpoint_info
        self._save_versions()

        if is_best:
            best_path = self.base_dir / "best_model.pth"
            torch.save(state, best_path)

        return checkpoint_path
