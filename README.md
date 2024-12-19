# Chest X-Ray Classification using Deep Learning

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
- Python 3.9+
- Poetry

### Installation
```bash
# Clone the repository
git clone https://github.com/pgryko/mnist-chest-xray-classification
cd mnist-chest-xray-classification

# Install dependencies using Poetry
poetry install
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
  - DenseNet
  - EfficientNet

### Model Interpretability
- Grad-CAM visualization
- SHAP values
- LIME explanations
- Integrated Gradients

### Experiment Tracking
- MLflow integration for experiment monitoring
- Hyperparameter optimization
- Performance metrics tracking
- Resource utilization monitoring

## 📊 Performance Metrics
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Classification Report

## 🔬 Model Training

### Custom CNN Training
```python
from src.training import Trainer
from src.models import ChestNetL

# Initialize model and trainer
model = ChestNetL()
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)

# Start training
trainer.train()
```

### Transfer Learning
```python
from src.models import PretrainedModel

# Initialize pretrained model
model = PretrainedModel(
    architecture='resnet50',
    pretrained=True,
    num_classes=1
)

# Fine-tune
trainer = Trainer(model=model, ...)
trainer.train()
```

## 📈 Visualization

```python
from src.utils.visualization import VisualizationManager

vis_manager = VisualizationManager(model, dataset)
vis_manager.plot_training_history(history)
vis_manager.plot_feature_maps(sample_image)
```

## 🔍 Model Interpretation

```python
from src.interpretability import ModelInterpreter

interpreter = ModelInterpreter(model)
interpretation = interpreter.explain(sample_image)
```

## 🚀 Deployment

### FastAPI Service
```bash
# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t chest-xray-classifier .

# Run container
docker run -p 8000:8000 chest-xray-classifier
```

## 📝 Documentation

### Model Architecture
Detailed documentation about model architectures, training procedures, and hyperparameter configurations can be found in `docs/`.

### API Documentation
API endpoints are documented using OpenAPI (Swagger) and can be accessed at `http://localhost:8000/docs` when running the service.

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


## 🤝 Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- MedMNIST dataset creators
- PyTorch team
- MLflow contributors

## 📞 Contact
For questions or feedback, please open an issue.

