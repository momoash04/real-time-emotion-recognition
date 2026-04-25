import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import mediapipe.python.solutions.face_detection as mp_face_detection_module

# --- 1. INITIALIZE FACE DETECTOR ---
mp_face_detection = mp_face_detection_module

# --- 2. MODEL CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


def load_model(model_path):
    # CRITICAL FIX: Must match your new Kaggle architecture (B2)
    model = models.efficientnet_b2()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 7)
    )

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


# Preprocessing for the face crop
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. LIVE DEMO WITH CROPPING ---
model = load_model('emotion_model_ULTIMATE_V2.pth')
cap = cv2.VideoCapture(0)

print(f"🚀 V2 Engine Started on: {device}")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            # We assume one main face for the dashboard
            detection = results.detections[0]

            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w_frame)
            y = int(bboxC.ymin * h_frame)
            w = int(bboxC.width * w_frame)
            h = int(bboxC.height * h_frame)

            x, y = max(0, x), max(0, y)
            x_end, y_end = min(w_frame, x + w), min(h_frame, y + h)
            w, h = x_end - x, y_end - y

            if w > 0 and h > 0:
                # 1. Extract & Preprocess
                face_roi = frame[y:y + h, x:x + w]
                roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # 2. Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, predicted = torch.max(probs, 0)
                    emotion = labels[predicted.item()]

                # 3. Main Bounding Box UI
                color_map = {
                    'Happiness': (0, 255, 0),
                    'Anger': (0, 0, 255),
                    'Surprise': (255, 255, 0),
                    'Neutral': (255, 255, 255),
                    'Fear': (128, 0, 128),  # Purple
                    'Disgust': (0, 165, 255),  # Orange
                    'Sadness': (255, 0, 0)  # Blue
                }
                color = color_map.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{emotion}: {conf * 100:.1f}%', (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 4. The Analytics Dashboard (Top Left)
                # Background panel
                cv2.rectangle(frame, (10, 10), (220, 190), (0, 0, 0), -1)
                cv2.putText(frame, "Live Analytics", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Draw dynamic bars
                for i, (label, prob) in enumerate(zip(labels, probs)):
                    bar_width = int(prob.item() * 100)
                    y_pos = 50 + i * 20

                    # Bar color matches the emotion map
                    bar_color = color_map.get(label, (255, 255, 255))

                    # Fill bar
                    cv2.rectangle(frame, (20, y_pos), (20 + bar_width, y_pos + 10), bar_color, -1)
                    # Text label
                    cv2.putText(frame, f'{label}: {prob.item() * 100:.0f}%', (125, y_pos + 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('V2 Emotion Detection Engine', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()