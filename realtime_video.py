import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import deque
import tkinter as tk
from tkinter import filedialog

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
MODEL_PATH = 'emotion_model_ULTIMATE_V2 (2).pth'

# ImageNet normalization from your notebook
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(path):
    # Reconstructing the EfficientNet-B2 architecture
    model = models.efficientnet_b2()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 7)
    )
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    print(f"🚀 Initializing Engine on {device}...")
    model = load_model(MODEL_PATH)
    mp_face = mp.solutions.face_detection

    # Define your maximum screen constraints here
    MAX_WIDTH = 1280
    MAX_HEIGHT = 720

    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")]
        )
        root.destroy()

        if not file_path:
            break

        is_image = file_path.lower().endswith(('.jpg', '.jpeg', '.png'))
        cap = cv2.VideoCapture(file_path)
        prob_history = deque(maxlen=5)

        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # --- NEW RESIZING LOGIC ---
                h_orig, w_orig, _ = frame.shape
                # Calculate the scaling factor to fit within MAX_WIDTH and MAX_HEIGHT
                scaling_factor = min(MAX_WIDTH / w_orig, MAX_HEIGHT / h_orig)

                # Only downscale if the image is larger than our constraints
                if scaling_factor < 1:
                    new_size = (int(w_orig * scaling_factor), int(h_orig * scaling_factor))
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

                h, w, _ = frame.shape  # Update dimensions to scaled size
                # --------------------------

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(img_rgb)

                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                        bw, bh = int(bbox.width * w), int(bbox.height * h)

                        face_roi = frame[max(0, y):y + bh, max(0, x):x + bw]
                        if face_roi.size == 0: continue

                        pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        input_tensor = transform(pil_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                            if not is_image:
                                prob_history.append(probs)
                                probs = torch.stack(list(prob_history)).mean(dim=0)

                            conf, pred = torch.max(probs, 0)
                            label = f"{labels[pred.item()]} {conf * 100:.1f}%"

                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Testing Model (Press Q to finish current file)', frame)

                if is_image:
                    cv2.waitKey(0)
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()