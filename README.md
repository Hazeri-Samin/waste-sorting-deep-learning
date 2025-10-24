
# Waste Classification — Image + Text (PyTorch)

Classify household waste into disposal categories from an image and a short description.  
This project combines a ResNet18 image backbone with a simple text encoder, then fuses both to predict the correct bin.

## 🏫 University of Calgary TALC HPC Environment
This project was developed and executed entirely on the **University of Calgary’s TALC HPC cluster**.  
All datasets were stored on secure university file systems.  
Due to access and licensing restrictions, the dataset is **not included** in this repository.

---

## Why it matters
Correct sorting improves recycling rates and reduces landfill contamination.  
A practical model like this can guide users in apps or kiosks.

---

## Project at a glance
- **Task:** 4-class classification → `Blue`, `Black`, `Green`, `TTr`
- **Modality:** Multimodal (image + text)
- **Model:** ResNet18 for image features + BiLSTM (or GRU) for text features + fusion head
- **Framework:** PyTorch
- **Training:** Adam optimizer, early stopping on validation loss
- **Evaluation:** Accuracy, F1, classification report, misclassified samples
- **Infrastructure:** TALC HPC cluster (University of Calgary)

---

## Repo structure
```text
.
├── notebooks/
│   └── DataAnalysis.ipynb            # main end-to-end workflow (executed on TALC)
├── src/
│   └── metrics/
│       ├── plot_confusion_matrix.py  # helper for confusion matrix plots
│       └── plot_training_curves.py   # helper for loss/accuracy curves
├── data/
│   └── README_DATA.txt               # explains dataset access
├── visuals/
│   ├── sample_predictions.png
│   ├── confusion_matrix.png
│   └── training_curves.png
├── .gitignore
└── README.md
```

---

## Dataset
- **Location:** TALC HPC storage (restricted)
- **Classes:** `Blue`, `Black`, `Green`, `TTr`
- **Modality:** RGB image + short text description per item
- **Source:** Stored on TALC (University of Calgary)
- **Availability:** ❌ Not public. Not distributed.
- **Access:** Students with HPC credentials only.

> If you wish to reproduce the project outside TALC, you must use your own dataset with the same class structure.

---

## Model summary
**Image branch**
- Backbone: `torchvision.models.resnet18` (pretrained)
- Output: pooled feature vector

**Text branch**
- Tokenization: simple vocab / torchtext (or your own tokenizer)
- Encoder: BiLSTM/GRU
- Output: pooled sequence embedding

**Fusion head**
- Concatenate image and text embeddings
- Fully connected layers + dropout
- Output: 4-way softmax

---

## Training details
- **Loss:** Cross-entropy
- **Optimizer:** Adam
- **Augmentation:** standard image transforms
- **Regularization:** dropout, early stopping on validation loss
- **Hardware:** TALC GPU/CPU nodes

---

## Results
- Final test accuracy: **TODO %**
- Macro F1: **TODO**
- Per-class report: see classification report output
- Error analysis: misclassified samples plotted

**Screenshots**
![Confusion Matrix](./visuals/confusion_matrix.png)
![Training Curves](./visuals/training_curves.png)
![Sample Predictions](./visuals/sample_predictions.png)

---

## How to run

### 1) TALC HPC
- Clone this repository into your TALC workspace.
- Load required modules or activate your Python environment (PyTorch, pandas, scikit-learn, etc.).
- Launch Jupyter on TALC or run the notebook directly.
- Paths in `DataAnalysis.ipynb` point to datasets stored in the HPC environment.

> ⚠️ **No dataset is provided in this repo**.  
> If reproducing outside TALC, supply your own dataset with identical structure and class labels.

### 2) Exporting Visuals
Inside your notebook:
```python
from src.metrics.plot_confusion_matrix import save_confusion_matrix
from src.metrics.plot_training_curves import save_training_curves
```
```python
save_confusion_matrix(y_true, y_pred, labels=['Blue','Black','Green','TTr'],
                      out_path='visuals/confusion_matrix.png')
save_training_curves(train_losses, val_losses, out_path='visuals/training_curves.png')
```

---

## What I learned
- Running ML experiments on an HPC environment (TALC)
- Managing multimodal datasets securely
- Multimodal fusion with PyTorch
- Error analysis and visualization in a restricted environment

---

## Next steps
- Integrate class weights or focal loss
- Experiment with transformer text encoders
- Model export with ONNX or TorchScript for deployment

---
