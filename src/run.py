"""
run_live.py
Real-time gesture recognition + sentence builder + text-to-speech.

Usage:
    python run_live.py

Controls:
    - Hold a sign steady for ~1 second (stability buffer) to confirm a letter.
    - Confirmed letters build up into a sentence on screen.
    - Press SPACE to add a space.
    - Press BACKSPACE to delete last character.
    - Press 's' to speak the current sentence aloud.
    - Press 'c' to clear.
    - Press 'q' to quit.
"""

import cv2
import joblib
import pyttsx3
from collections import deque, Counter
from utils import create_hand_detector, normalize_landmarks, draw_landmarks

MODEL_PATH = "../models/gesture_model.pkl"
STABILITY_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.6


def speak_text(text):
    """
    Speak the provided text.

    A new pyttsx3 engine is created each time so that
    TTS can be triggered repeatedly.
    """
    if not text.strip():
        print("Nothing to speak.")
        return

    try:
        print(f"Speaking: {text}")

        # Create a fresh engine for every speech request
        engine = pyttsx3.init()

        # Speech settings
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)

        # Speak
        engine.say(text)
        engine.runAndWait()

        # Properly shut down the engine
        engine.stop()

    except Exception as e:
        print(f"TTS error: {e}")


def main():
    # Load trained gesture model
    model = joblib.load(MODEL_PATH)

    # Create hand detector
    detector = create_hand_detector(max_hands=1)

    # Camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Prediction stability buffer
    prediction_buffer = deque(maxlen=STABILITY_FRAMES)

    # Current sentence
    sentence = ""

    # Last confirmed letter
    last_confirmed = None

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Mirror camera image
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        result = detector.process(rgb)

        current_label = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw hand landmarks
            draw_landmarks(frame, hand_landmarks)

            # Normalize landmarks
            coords = normalize_landmarks(hand_landmarks)

            # Get model probabilities
            probs = model.predict_proba([coords])[0]

            # Get highest probability
            best_idx = probs.argmax()
            confidence = probs[best_idx]

            # Get predicted letter
            predicted_label = model.classes_[best_idx]

            # Only accept prediction if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                current_label = predicted_label

                cv2.putText(
                    frame,
                    f"{predicted_label} ({confidence:.2f})",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        # Add prediction to stability buffer
        prediction_buffer.append(current_label)

        # Confirm a letter only when all recent frames agree
        if len(prediction_buffer) == STABILITY_FRAMES:

            most_common, count = Counter(
                prediction_buffer
            ).most_common(1)[0]

            # Confirm letter
            if (
                most_common is not None
                and count == STABILITY_FRAMES
                and most_common != last_confirmed
            ):
                sentence += most_common

                last_confirmed = most_common

                # Clear buffer so another letter can be detected
                prediction_buffer.clear()

            # If no hand/prediction, allow the same letter
            # to be detected again later
            elif most_common is None:
                last_confirmed = None

        # Display sentence
        cv2.putText(
            frame,
            f"Sentence: {sentence}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display controls
        cv2.putText(
            frame,
            "SPACE=space BKSP=delete S=speak C=clear Q=quit",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # Show camera window
        cv2.imshow(
            "Sign Language Recognizer",
            frame
        )

        # Get keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q') or key == ord('Q'):
            break

        # Add space
        elif key == ord(' '):
            sentence += " "

        # Backspace
        elif key == 8:
            sentence = sentence[:-1]

        # Speak
        elif key == ord('s') or key == ord('S'):
            speak_text(sentence)

        # Clear sentence
        elif key == ord('c') or key == ord('C'):
            sentence = ""
            prediction_buffer.clear()
            last_confirmed = None

    # Release camera
    cap.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
