# Deepfake Detector

Deepfake image classification project built with PyTorch.

The project explores a hybrid architecture that combines a pretrained ResNet18 feature extractor with an LSTM and an attention mechanism for binary classification of images into real and fake classes.

## Model Architecture

Input Image → ResNet18 → Feature Map → LSTM → Attention → Fully Connected Layer → Real / Fake

## Technologies

- Python
- PyTorch
- torchvision
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- Google Colab

## Features

- transfer learning with pretrained ResNet18
- image augmentation
- LSTM-based feature processing
- attention mechanism
- early stopping
- learning rate scheduling
- gradient clipping
- evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
- confusion matrix
- ROC curve
- Grad-CAM visualization

## Results

Approximate validation results:

- Accuracy: ~76%
- ROC-AUC: ~0.79
- Real class recall: ~0.94
- Fake class recall: ~0.42

## Dataset

The project uses a relatively small dataset of approximately 479 images.

Because of the limited dataset size, the reported metrics should be treated as experimental results.

The dataset is not included in this repository.

## Limitations

- small dataset size
- random train/validation split
- possible similarity between training and validation images
- no external test dataset
- model is not intended for production use

## Future Improvements

- larger dataset
- identity-based train/validation split
- comparison with a simple ResNet18 baseline
- testing EfficientNet or Vision Transformer
- improving fake-class recall
- external dataset evaluation

## Notebook

Main implementation:

`deepfake_detector.ipynb`

## Status

Educational and experimental computer vision project.
