# Chest X-Ray Classification using Deep Learning and MLFLOW

## Project Overview
This project implements a deep learning solution for binary classification of chest X-rays using the ChestMNIST dataset from MedMNIST. The system includes custom CNN architectures, transfer learning approaches, and extensive model interpretability features.

## 🔍 Domain Knowledge
Chest X-ray analysis is a critical diagnostic tool in pulmonary medicine. The binary classification task focuses on detecting abnormalities in chest X-rays, which can indicate various conditions including:
- Pneumonia
- Pulmonary edema
- Cardiomegaly
- Pleural effusions

## 🏗️ Project Structure
```
chest-xray-classification/
├── pyproject.toml          # Poetry dependency management
├── README.md
├── src/
│   ├── models/            # Model architectures
│   │   ├── custom_cnn.py
│   │   └── pretrained.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── data/              # Data handling
│   │   ├── dataset.py
│   │   └── augmentation.py
│   ├── utils/             # Utility functions
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── interpretability/   # Model interpretation
│       ├── gradcam.py
│       └── shap_explainer.py
├── tests/                 # Unit tests
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
└── docs/                # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Poetry

### Installation
```bash
# Clone the repository
git clone https://github.com/pgryko/mnist-chest-xray-classification
cd mnist-chest-xray-classification

# Install dependencies using Poetry
poetry install
```



#### Install Git LFS
We use git-lfs to store large files such as model weights and datasets. 
To install Git LFS, follow the instructions below:

For Ubuntu/Debian
```bash
sudo apt-get install git-lfs
```

For macOS using Homebrew
```bash
brew install git-lfs
```

Initialize Git LFS
```bash
git lfs install
```


### Dataset Setup
```python
from medmnist import ChestMNIST

# Download and prepare dataset
train_dataset = ChestMNIST(split='train', download=True)
val_dataset = ChestMNIST(split='val', download=True)
test_dataset = ChestMNIST(split='test', download=True)
```

## 🛠️ Features

### Custom CNN Architectures
- ChestNet-S: Lightweight architecture for quick experimentation
- ChestNet-M: Medium-sized network with residual connections
- ChestNet-L: Large architecture with attention mechanisms

### Transfer Learning
- Fine-tuning options for popular architectures:
  - ResNet

[//]: # (### Model Interpretability)

[//]: # (- Grad-CAM visualization)

[//]: # (- SHAP values)

[//]: # (- LIME explanations)

[//]: # (- Integrated Gradients)

### Experiment Tracking
- MLflow integration for experiment monitoring
- Performance metrics tracking
- Resource utilization monitoring

## 📊 Performance Metrics
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Classification Report

### MLflow Tracking
```bash
mlflow server --host 127.0.0.1 --port 5000
```


## 🧪 Testing
```bash
# Run tests
poetry run pytest
```

## 📊 Results
- Training Metrics
- Validation Results
- Test Set Performance
- Model Interpretability Insights

```

To use presentationw:

1. Install Marp CLI:
```bash
npm install -g @marp-team/marp-cli
```

2. Convert to PDF/PPTX:
```bash
marp --pdf presentation.md
# or
marp --pptx presentation.md
```


## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- MedMNIST dataset creators
- PyTorch team
- MLflow contributors

## 📞 Contact
For questions or feedback, please open an issue.

