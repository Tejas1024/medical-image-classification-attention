# Attention-Enhanced Ensemble CNN for Medical Image Classification

## Overview

This project presents a novel deep learning framework for medical image classification using attention-enhanced CNN ensemble learning. The system combines multi-scale feature extraction, channel and spatial attention mechanisms, and ensemble learning with uncertainty quantification to achieve state-of-the-art performance on medical imaging tasks.

## Architecture

### Core Components
1. **Multi-backbone Ensemble**: ResNet50, EfficientNetB0, DenseNet121
2. **Attention Mechanism**: Channel and spatial attention blocks
3. **Ensemble Learning**: Voting-based prediction aggregation
4. **Uncertainty Quantification**: Entropy-based prediction uncertainty

### Key Features
- Pre-trained backbone networks
- Custom attention layers
- Data augmentation (RandomFlip, RandomRotation)
- Early stopping and regularization
- Comprehensive evaluation metrics
- Visualization pipelines

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from main import MedicalImageClassifier

# Initialize classifier
classifier = MedicalImageClassifier(input_shape=(224, 224, 3), num_classes=3)

# Generate or load data
X, y = classifier.generate_synthetic_data(num_samples=600)

# Train ensemble
classifier.train(X_train, y_train, X_val, y_val, epochs=20)

# Evaluate
results = classifier.evaluate(X_test, y_test)

# Visualize
classifier.visualize_results(results)
```

## Training Configuration

- **Input Shape**: 224x224x3
- **Batch Size**: 32
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Early Stopping**: Patience=3

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curves
- Confusion Matrix
- Prediction Uncertainty

## Results

All results, visualizations, and reports are saved to the `results/` directory:
- `ensemble_evaluation.png`: Comprehensive evaluation plots
- Model checkpoints for each backbone
- Classification reports

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
3. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks.
4. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks.

## License

MIT License
