# Real-Time Sign Language Gesture Recognizer

A webcam-based fingerspelling recognizer. Hold up a hand sign, and the system recognizes the letter or digit, builds words and sentences on screen, auto-corrects spelling, and reads the result aloud with text-to-speech.

Built with Python, OpenCV, MediaPipe, and scikit-learn.

## How it works

```
Webcam frame (OpenCV)
   -> 21 hand landmarks (MediaPipe Hands)
      -> wrist-relative, scale-normalized features (42 values)
         -> Random Forest classifier (scikit-learn)
            -> stability buffer (letter must hold for N frames)
               -> sentence builder + spell-check
                  -> text-to-speech (pyttsx3)
```

1. **Detect**: MediaPipe returns 21 (x, y) landmarks for the hand in each frame.
2. **Normalize**: Coordinates are shifted so the wrist is the origin and scaled by the largest absolute value. The same sign gives similar features regardless of where the hand is in the frame or how far it is from the camera.
3. **Classify**: A Random Forest trained on self-collected landmark data predicts the sign, with a confidence threshold to reject uncertain frames.
4. **Stabilize**: A letter is only confirmed after the same prediction holds across a window of consecutive frames, which removes flicker between signs.
5. **Build and speak**: Confirmed letters form words. Pressing space triggers dictionary-based autocorrect on the finished word, and the full sentence can be spoken aloud.

## Features

- Recognizes 36 classes: letters A-Z and digits 0-9
- Landmark-based features, so no images are stored and the dataset is a small CSV
- Confidence thresholding and frame-stability confirmation
- On-screen sentence builder with space, backspace, and clear
- Optional spell-check on each finished word
- Text-to-speech output
- Modular three-stage pipeline: collect, train, run

## Project structure

```
sign-language-recognizer/
├── data/
│   └── gestures.csv          # collected landmark dataset (label + 42 features)
├── models/
│   └── gesture_model.pkl     # trained Random Forest (generated)
├── src/
│   ├── utils.py              # hand detector, landmark normalization, drawing
│   ├── collect_data.py       # data collection tool
│   ├── train_model.py        # training + evaluation
│   └── run.py              # real-time recognition and speech
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Requires **Python 3.10** (MediaPipe's `solutions` API is used, so the version pin below matters).

```bash
git clone https://github.com/khizerdotpy/sign-language-recognizer.git
cd sign-language-recognizer

# Windows
py -3.10 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `mediapipe` is pinned to `0.10.21`. Newer releases removed `mediapipe.solutions`, which this project depends on.

## Usage

All scripts use relative paths, so run them from inside `src/`.

```bash
cd src
```

### 1. Collect data

```bash
python collect_data.py
```

Hold a sign, then press the matching key (a-z, 0-9) to save one sample. Press `q` to quit. For a robust dataset, vary distance, position, rotation, and lighting between presses, and collect in several separate sessions.

### 2. Train

```bash
python train_model.py
```

Prints accuracy, a per-class report, and a confusion matrix, then saves the model to `models/gesture_model.pkl`.

### 3. Run live

```bash
python run.py
```

| Key | Action |
|---|---|
| Hold a sign | Confirms a letter after a stable window of frames |
| `SPACE` | Finish the current word |
| `BACKSPACE` | Delete last character |
| `S` | Speak the sentence |
| `C` | Clear the sentence |
| `Q` | Quit |

## Dataset

- 10,131 samples across 36 classes (about 270-300 per class)
- Each sample is a row of 42 normalized landmark coordinates with a label
- Self-collected with a single webcam

## Results

- Held-out test accuracy (random split):99.95%

## Limitations

- **Static signs only.** J and Z are motion-based in ASL and and as this project requires sign identification from a single frame, static versions for those two classes were used.
- **Single-person, single-setup data.** The model was trained on one person's hand under limited conditions, so a high random-split accuracy can overstate real-world performance. Merging in a public dataset or adding more varied sessions is the planned fix.
- **One hand.** Landmarks are not mirror-corrected, so a model trained on one hand may not transfer to the other.
- **Fingerspelling only.** Recognizes the manual alphabet and digits, not full ASL words or phrases.


## Tech stack

Python 3.10 · OpenCV · MediaPipe · scikit-learn · pandas · NumPy · joblib · pyttsx3 · pyspellchecker

## License

MIT