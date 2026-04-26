import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import time
import mediapipe.python.solutions.face_detection as mp_face_detection_module

# --- 1. INITIALIZE FACE DETECTOR ---
mp_face_detection = mp_face_detection_module

# --- 2. MODEL CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


def load_model(model_path):
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


# --- GUI HELPER FUNCTION ---
def draw_smart_tracking_box(img, x, y, w, h, color, thickness=3, length=30):
    """Draws a modern corner-bracket bounding box instead of a plain square."""
    # Top-Left
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)


# --- 3. LIVE DEMO WITH FULLSCREEN GUI ---
model = load_model('emotion_model_ULTIMATE_V2 (2).pth')  # Make sure name matches!

# Initialize Camera and request High Definition
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Setup Fullscreen Window
# Setup Maximized Window (keeps the X button and title bar)
window_name = 'V2 Emotion Detection Engine'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# We delete the WND_PROP_FULLSCREEN line!
# Instead, we force the window to open at a nice, large 720p size:
cv2.resizeWindow(window_name, 1280, 720)

print(f"🚀 V2 Engine Started on: {device}")
prev_time = 0

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
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
                    'Fear': (128, 0, 128),
                    'Disgust': (0, 165, 255),
                    'Sadness': (255, 0, 0)
                }
                color = color_map.get(emotion, (255, 255, 255))

                # Draw sleek tracking box instead of standard rectangle
                draw_smart_tracking_box(frame, x, y, w, h, color, thickness=3)

                # Add a stylish label background
                label_text = f'{emotion} {conf * 100:.1f}%'
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
                # Draw text in black over the colored background for contrast
                cv2.putText(frame, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # 4. The Analytics Dashboard (Semi-Transparent Overlay)
                overlay = frame.copy()
                # Draw the black panel on the overlay
                cv2.rectangle(overlay, (20, 20), (320, 310), (0, 0, 0), -1)

                # Blend the overlay with the original frame (Alpha = 0.6)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Add Dashboard Headers
                cv2.putText(frame, "AI EMOTION ANALYTICS", (35, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                cv2.line(frame, (35, 70), (280, 70), (255, 255, 255), 1)

                # Draw larger, dynamic bars
                for i, (label, prob) in enumerate(zip(labels, probs)):
                    # Scale bar width to fit the new larger UI (max 150 pixels)
                    bar_width = int(prob.item() * 150)
                    y_pos = 90 + i * 30

                    bar_color = color_map.get(label, (255, 255, 255))

                    # Fill bar (Thicker)
                    cv2.rectangle(frame, (35, y_pos), (35 + bar_width, y_pos + 15), bar_color, -1)

                    # Draw subtle border for empty space
                    cv2.rectangle(frame, (35, y_pos), (185, y_pos + 15), (100, 100, 100), 1)

                    # Text label (Larger)
                    cv2.putText(frame, f'{label}: {prob.item() * 100:.0f}%', (195, y_pos + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 5. Global UI Elements (FPS Counter)
        cv2.putText(frame, f'FPS: {int(fps)}', (w_frame - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        # Press 'q' to quit fullscreen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
