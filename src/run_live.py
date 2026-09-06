"""
run_live.py
Real-time gesture recognition + sentence builder + text-to-speech.

Usage:
    python run_live.py

Controls:
    - Hold a sign steady for ~1 second (stability buffer) to confirm a letter.
    - Confirmed letters build up into a sentence on screen.
    - Press SPACE to add a space, BACKSPACE to delete last char,
      's' to speak the current sentence aloud, 'c' to clear, 'q' to quit.
"""

import cv2
import joblib
import pyttsx3
from collections import deque, Counter
from utils import create_hand_detector, normalize_landmarks, draw_landmarks

MODEL_PATH = "../models/gesture_model.pkl"
STABILITY_FRAMES = 15   # how many consistent frames needed to confirm a letter
CONFIDENCE_THRESHOLD = 0.6


def main():
    model = joblib.load(MODEL_PATH)
    detector = create_hand_detector(max_hands=1)
    tts_engine = pyttsx3.init()

    cap = cv2.VideoCapture(0)

    prediction_buffer = deque(maxlen=STABILITY_FRAMES)
    sentence = ""
    last_confirmed = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        current_label = None
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            coords = normalize_landmarks(hand_landmarks)

            probs = model.predict_proba([coords])[0]
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            predicted_label = model.classes_[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                current_label = predicted_label
                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        prediction_buffer.append(current_label)

        # Confirm a letter only if it dominates the recent buffer (stability check)
        if len(prediction_buffer) == STABILITY_FRAMES:
            most_common, count = Counter(prediction_buffer).most_common(1)[0]
            if most_common is not None and count == STABILITY_FRAMES and most_common != last_confirmed:
                sentence += most_common
                last_confirmed = most_common
                prediction_buffer.clear()
            elif most_common is None:
                last_confirmed = None

        cv2.putText(frame, f"Sentence: {sentence}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=space BKSP=delete s=speak c=clear q=quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Language Recognizer", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            sentence += " "
        elif key == 8:  # backspace
            sentence = sentence[:-1]
        elif key == ord('c'):
            sentence = ""
        elif key == ord('s'):
            tts_engine.say(sentence)
            tts_engine.runAndWait()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
