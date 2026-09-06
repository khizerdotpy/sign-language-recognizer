import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hand_detector(max_hands=1, detection_conf=0.7, tracking_conf=0.6):
    #hand helper instance
    return mp_hands.Hands(
        max_num_hands=max_hands,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )


def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe's 21 landmarks (absolute, image-relative x,y)
    into a 42-length list of wrist-relative, scale-normalized coordinates.

    Why:
    - Subtracting the wrist (landmark 0) makes the representation
      translation-invariant (doesn't matter WHERE in the frame the hand is).
    - Dividing by the max absolute value makes it scale-invariant
      (doesn't matter how close/far the hand is from the camera).
    """
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)

    max_val = max(max(abs(c) for c in coords), 1e-6)
    coords = [c / max_val for c in coords]
    return coords  # length 42: [x0,y0,x1,y1,...,x20,y20]


def draw_landmarks(frame, hand_landmarks):
    """Draws the hand skeleton on the frame for visual debugging."""
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
