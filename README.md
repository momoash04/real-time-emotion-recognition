# Real-Time Emotion Recognition

An end-to-end deep learning project that detects facial emotions in real-time from webcams, video files, and images. This project leverages an EfficientNet-B2 architecture trained on a custom hybrid dataset, utilizing advanced techniques like dataset injection, Focal Loss, and Stochastic Weight Averaging (SWA) to achieve robust performance.

## Table of Contents
- [Overview](#overview)
- [Training Phase in Detail](#training-phase-in-detail)
  - [Dataset Fusion & Injection](#dataset-fusion--injection)
  - [Model Architecture](#model-architecture)
  - [Focal Loss](#focal-loss)
  - [Two-Stage Training & SWA](#two-stage-training--swa)
- [Inference & Setup](#inference--setup)
  - [Real-Time Webcam](#real-time-webcam)
  - [Video/Image File Analysis](#videoimage-file-analysis)
- [Requirements](#requirements)

## Overview
This system is capable of detecting 7 basic emotions (`Surprise`, `Fear`, `Disgust`, `Happiness`, `Sadness`, `Anger`, `Neutral`). It uses Google's **MediaPipe** for ultra-fast face detection and a highly optimized PyTorch model for emotion classification. The inference scripts feature custom UI elements like smoothed probability bars and modern bounding boxes.

---

## Training Phase in Detail
The training process is documented in `kaggle_train_notebook.ipynb`. It outlines a rigorous pipeline designed to combat class imbalance and boost model accuracy.

### Dataset Fusion & Injection
One of the biggest challenges in emotion recognition is class imbalance (e.g., having too many 'Happy' faces and not enough 'Disgust' or 'Fear').
- **Base Dataset:** We use the high-quality **RAF-DB** dataset as the foundation.
- **FER2013+ Injection:** To solve the imbalance, we inject targeted samples from the **FER2013+** dataset into the minority classes (`Fear`, `Disgust`, and `Anger`). By limiting the injection to ~800 images per class, we successfully balance them to be on par with the majority classes, creating a robust **FUSED_DATASET**.
- **Data Augmentation:** The training pipeline applies heavy data augmentation (Random Horizontal Flip, Rotation, ColorJitter, RandomAffine, and RandomErasing) to prevent overfitting and improve generalization to unseen faces.

### Model Architecture
- The model uses an **EfficientNet-B2** backbone (pretrained on ImageNet).
- The default classifier head is replaced with a custom `Sequential` block featuring a 512-neuron Hidden layer, ReLU activation, 40% Dropout for regularization, and a final 7-neuron output layer corresponding to our emotions.

### Focal Loss
Instead of standard Cross-Entropy, the model is trained using **Focal Loss** (`gamma=2`, with label smoothing). Focal Loss dynamically scales the loss based on prediction confidence, forcing the model to focus heavily on hard-to-classify examples (like subtle expressions of Fear or Disgust) rather than easily classified ones.

### Two-Stage Training & SWA
1. **Phase 1 (Head Training):** The EfficientNet-B2 backbone is frozen, and only the new classifier head is trained for 10 epochs at a higher learning rate (`0.001`).
2. **Phase 2 (Global Fine-Tuning):** The entire model is unfrozen and fine-tuned at a lower learning rate (`1e-5`) for 35 epochs. 
3. **Stochastic Weight Averaging (SWA):** During the final epochs of Phase 2, SWA is applied. This technique averages the model weights over the last few epochs, leading to a flatter minimum in the loss landscape and significantly improving the model's ability to generalize to new, unseen faces.

---

## Inference & Setup

Both inference scripts require the trained model file (`emotion_model_ULTIMATE_V2 (2).pth`) to be in the same directory. They use **MediaPipe Face Detection**, which provides incredibly fast and accurate face cropping before passing the region of interest (ROI) to the PyTorch model.

### Real-Time Webcam
Run the webcam demo:
```bash
python realtime_webcam.py
```
**Features:**
- **Temporal Smoothing:** To prevent the predictions from aggressively flickering between frames, a queue (`deque`) stores the probability tensors of the last 5 frames. The displayed emotion is the average of these probabilities, resulting in a buttery-smooth UI.
- **Smart UI:** Features a sleek "corner-bracket" bounding box, a semi-transparent analytics dashboard overlay, and dynamic probability bars for all 7 emotions.
- **FPS Counter:** Real-time performance tracking.

### Video/Image File Analysis
Analyze existing media files:
```bash
python realtime_video.py
```
**Features:**
- **File Picker:** Uses a `Tkinter` file dialog to let you easily browse and select `.jpg`, `.png`, `.mp4`, `.avi`, etc.
- **Dynamic Resizing:** Automatically scales down high-resolution videos/images to fit within a `1280x720` constraint for optimal viewing without stretching.
- **Smoothing:** Applies the same temporal smoothing technique to videos (disabled for single images).

## Requirements
Ensure you have the following installed:
- `torch` and `torchvision`
- `opencv-python` (cv2)
- `mediapipe`
- `numpy`
- `Pillow` (PIL)
- `scikit-learn`, `matplotlib`, `seaborn` (for training analytics)
