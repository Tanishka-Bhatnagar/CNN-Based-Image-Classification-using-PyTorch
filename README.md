An end-to-end CNN image classification system build from scratch using **PyTorch** and deployed as an interactive web app via **Gradio** and **Hugging Face Spaces**.

Project live at:
https://huggingface.co/spaces/Tanishka111/animal-face-classifier

## Problem
Organizations increasingly rely on automated visual inspection systems to classify images and identify objects accurately. Manual image classification is slow, repetitive, and difficult to scale—especially when images contain variations in lighting, orientation, quality, and object appearance.

## Objective: build and deploy a PyTorch-based CNN pipeline
- Preprocessing raw image data into standardized tensors.
- Automatic extraction of visual features like edges, textures and facial structures.
- Accurate classification of unseen images into 3 categories: **Cat**, **Dog** and **Wild**.
- Interactive web interface for real time user interaction.

## Approach and Architecture
**Dataset** : AFHQ (Animal Faces-HQ) consist of 16,130 high resolution images pre split into independent `train` (14,630) and `val` (1500) having 3 balanced classes [`cat`, `dog`, `wild`].

**Preprocessing** : All images converted to 3-channel RGB, resized to `128x128` pixels for computational efficiency and normalized form `[0, 255]` pixel intensities into `[0.0, 1.0]` floating point tensor.

**CNN Architecture**
Divided into 2 stages `Feature Extraction` and `Fully Connected Layer for Classification`.
1. **Feature Extraction** : 3 Convolutional Blocks
   - **Block 1:** `Conv2d(3 → 32, kernel=3, padding=1)` + `ReLU` + `MaxPool2d(2×2)` → Output: `32 × 64 × 64`
   - **Block 2:** `Conv2d(32 → 64, kernel=3, padding=1)` + `ReLU` + `MaxPool2d(2×2)` → Output: `64 × 32 × 32`
   - **Block 3:** `Conv2d(64 → 128, kernel=3, padding=1)` + `ReLU` + `MaxPool2d(2×2)` → Output: `128 × 16 × 16`
2. **Classification** :
   - `Flatten()` → Converts `128 × 16 × 16` feature maps into a `32,768`-dimensional vector.
   - `Linear(32768 → 128)` + `ReLU` → Learns high-level feature combinations.
   - `Linear(128 → 3)` → Outputs raw class logits for `Cat`, `Dog`, and `Wild`.

## Training Strategy and Results
- **Loss Function:** `CrossEntropyLoss` (combines `LogSoftmax` and `NLLLoss` for multi-class classification).
- **Optimizer:** `Adam` (`lr = 0.001`) trained for `10 epochs` with a batch size of `32`.
- **Performance Metrics:**
  * **Training Accuracy:** `99.62%`
  * **Validation (Unseen Test) Accuracy:** `96.80%`
  * **Generalization Gap:** `2.82%` (demonstrating strong feature learning without severe overfitting).

## Project Structure

```text
├── app.py                     # Gradio web application & PyTorch inference pipeline
├── animal_cnn_weights.pth     # Saved trained model weights
├── requirements.txt           # Python dependencies
├── training_notebook.ipynb    # Google Colab notebook for preprocessing, training & evaluation
└── README.md                  # Project documentation
```

## Setup & Installation Instructions
* Python 3.9 or higher
* `pip` package manager
* install required libraries: pip install torch torchvision gradio pillow numpy

## How to run
Deployed on Hugging Face Space so can be directly accessed from link below: 
https://huggingface.co/spaces/Tanishka111/animal-face-classifier

Note: upload clear image of cat, dog or wild animal like lion, tiger, wolf.

## how to run locally
keep `app.py` and `animal_cnn_weights.pth` in same folder then run:
command-> python app.py
Open the local URL displayed in terminal

## Future improvements
* Domain Generalization & Fine-Tuning: fine tune existing weights with low learning rate (to avoid replacing previously learned patterns) on uncropped, full-body outdoor animal dataset using data augmentation (`RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`) to improve robustness on wide-angle user photos.
* Out-of-Distribution Handling:** Introduce a confidence threshold filter to flag non-animal uploads.
