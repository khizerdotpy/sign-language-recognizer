"""
collect_data.py
Webcam-based data collection tool.

Usage:
    python collect_data.py

Controls:
    - Hold a hand sign steady in front of the camera.
    - Press any letter/digit key (a-z, 0-9) to save the CURRENT normalized
      landmark row, labeled with that key, into data/gestures.csv.
    - Press 'q' to quit.

Tip: collect at least 60-100 samples per gesture, varying hand angle,
distance from camera, and lighting slightly each time. This is what
makes the model generalize instead of memorizing one exact hand pose.
"""

import csv
import os
import cv2
from utils import create_hand_detector, normalize_landmarks, draw_landmarks

DATA_PATH = "../data/gestures.csv"


def ensure_csv_header():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        header = ["label"] + [f"c{i}" for i in range(42)]
        with open(DATA_PATH, "w", newline="") as f:
            csv.writer(f).writerow(header)


def save_sample(label, coords):
    with open(DATA_PATH, "a", newline="") as f:
        csv.writer(f).writerow([label] + coords)


def main():
    ensure_csv_header()
    detector = create_hand_detector(max_hands=1)
    cap = cv2.VideoCapture(0)
    sample_count = 0

    print("Collecting data. Press a letter/digit key to label current pose. 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        current_coords = None
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            current_coords = normalize_landmarks(hand_landmarks)

        cv2.putText(frame, f"Samples saved: {sample_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press a-z/0-9 to label, q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif current_coords is not None and (chr(key).isalnum() if key < 256 else False):
            label = chr(key)
            save_sample(label, current_coords)
            sample_count += 1
            print(f"Saved sample for label '{label}' (total: {sample_count})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
