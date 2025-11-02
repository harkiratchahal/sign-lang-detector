# Sign Language Detector - Enhanced Web Application

A modern, interactive web application for creating custom sign language recognition systems with real-time classification and sentence building capabilities.

## ✨ New Features

- **Custom Sign Training**: Record and label your own signs with descriptive names
- **Real-time Classification**: Live webcam-based sign language recognition
- **Sentence Builder**: Chain multiple signs together to form complete sentences
- **Beautiful UI**: Modern, responsive interface with smooth animations
- **Easy Setup**: Automated virtual environment setup and dependency management
- **MediaPipe Integration**: Accurate hand landmark detection
- **FastAPI Backend**: Modern web framework with WebSocket support

---

## 🚀 Quick Start

### 1. Setup Virtual Environment

Run the setup script to create a virtual environment and install all dependencies:

```powershell
.\setup_venv.ps1
```

This will:

- Create a Python virtual environment in `venv/`
- Install all required packages from `requirements.txt`
- Set up the project for first use

### 2. Run the Application

Start the web application:

```powershell
.\run_app.ps1
```

The application will start at `http://localhost:8000`

Open your browser and navigate to the URL to access the web interface.

---

## 📖 How to Use the Web Application

### Recording Custom Signs

1. Navigate to the **Record Signs** tab
2. Enter a sign name (e.g., "hello", "thanks", "A")
3. Enter a description of what the sign means
4. Click **Add New Sign**
5. Click **Start Camera** to activate your webcam
6. Click **Start Capturing** to record samples (default: 100 samples)
7. Perform your sign repeatedly while the system captures images

### Training the Model

1. Navigate to the **Train Model** tab
2. Review your data statistics (number of signs and samples)
3. Click **Train Model** to train the AI
4. Wait for training to complete (typically 10-30 seconds)
5. Check the accuracy score

### Live Classification

1. Navigate to the **Live Classification** tab
2. Click **Start Classification**
3. Perform signs in front of your webcam
4. See real-time predictions with confidence scores

### Building Sentences

1. Navigate to the **Sentences** tab
2. Click **Start Recognition**
3. Perform a sign
4. Click **Add Current Sign** to add it to your sentence
5. Repeat to build complete sentences
6. Click **Clear Sentence** to start over

---

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Computer Vision**: MediaPipe, OpenCV
- **Machine Learning**: scikit-learn (RandomForest)
- **Real-time Communication**: WebSockets

---

## 📁 Project Structure

```
sign-language-detector/
├── app.py                  # FastAPI application
├── collect_imgs.py         # Image collection script (legacy)
├── create_dataset.py       # Dataset creation with MediaPipe
├── train_classifier.py     # Model training script
├── inference_classifier.py # Real-time inference (legacy)
├── requirements.txt        # Python dependencies
├── setup_venv.ps1         # Virtual environment setup script
├── run_app.ps1            # Application launcher script
├── data/                  # Training images (organized by label)
├── models/                # Trained models and data files
│   ├── model.pickle       # Trained classifier
│   ├── data.pickle        # Processed dataset
│   └── labels.json        # Label mappings
└── static/                # Web frontend
    ├── index.html         # Main HTML page
    ├── styles.css         # Styles and animations
    └── app.js             # Frontend JavaScript logic
```

---

## 🔧 Configuration

### Camera Settings

The application uses your default webcam. To change camera settings, modify the video constraints in `static/app.js`:

```javascript
navigator.mediaDevices.getUserMedia({
  video: { width: 1280, height: 720 },
});
```

### Training Parameters

Adjust training parameters in `train_classifier.py`:

```python
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    random_state=42,
    n_jobs=-1         # Use all CPU cores
)
```

---

## 📊 Data Collection Tips

For best results:

- Collect at least 50-100 samples per sign
- Use consistent lighting
- Vary hand positions slightly for robustness
- Use a plain background when possible
- Ensure your full hand is visible in the frame

---

## 🐛 Troubleshooting

### Camera Not Working

- Ensure no other application is using your webcam
- Grant browser permission to access the camera
- Try a different browser (Chrome/Edge recommended)

### Import Errors

Make sure the virtual environment is activated:

```powershell
.\venv\Scripts\Activate.ps1
```

Then reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### MediaPipe Installation Issues (Windows)

Install Visual C++ Build Tools if you encounter errors:

- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Or try: `pip install mediapipe==0.10.0`

### Low Accuracy

- Collect more training samples (100+ per sign)
- Ensure consistent hand positioning
- Retrain the model after collecting more data
- Check that signs are visually distinct

---

## 🚀 Advanced Usage

### Using Standalone Python Scripts

You can still use the original command-line scripts:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Create dataset from images
python create_dataset.py

# Train model
python train_classifier.py

# Run inference
python inference_classifier.py
```

### API Endpoints

The FastAPI backend exposes REST and WebSocket endpoints:

- `GET /api/status` - Check server and model status
- `GET /api/labels` - Get all labels
- `POST /api/labels` - Add new label
- `DELETE /api/labels/{label}` - Delete label
- `POST /api/capture` - Capture training image
- `POST /api/train` - Train model
- `GET /api/data/stats` - Get data statistics
- `WS /ws/inference` - WebSocket for real-time inference

---

## Old Documentation (Original Pipeline)

This repository contains a small computer-vision pipeline that collects hand images, extracts hand landmarks using MediaPipe, trains a Random Forest classifier and runs real-time inference from a webcam.

Contents

- `collect_imgs.py` — capture images from a webcam into `./data/<class_id>/` folders.
- `create_dataset.py` — process images in `./data` with MediaPipe to extract hand landmarks and save them to `data.pickle`.
- `train_classifier.py` — load `data.pickle`, train a RandomForest classifier and save `model.pickle`.
- `inference_classifier.py` — run webcam inference using MediaPipe and the trained model to display predicted labels in real time.
- `data/` — folder where captured images should be stored (ignored by git).

Quick overview (contract)

- Input: webcam frames or images placed under `data/<class_name_or_id>/*.jpg`.
- Output: `data.pickle` (features + labels) and `model.pickle` (trained model).
- Errors: missing camera, missing `data.pickle` or `model.pickle`, MediaPipe dependency issues.

Requirements

- Python 3.8+ recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

What each script does and how to use it

1. collect_imgs.py — create your dataset

- Purpose: capture images for each class (gesture/letter). The script creates folders `./data/0`, `./data/1`, ... up to `number_of_classes - 1` and saves `dataset_size` images per class.
- How to run: open a terminal and run `python collect_imgs.py`.
- How it works: it opens your default webcam, lets you preview frames and press `q` to start capturing, then saves sequential JPGs.
- Customize: change `number_of_classes` and `dataset_size` at the top of the file. You can also replace numeric class folders with named folders (e.g., `A`, `B`, `L`) — if you do, `create_dataset.py` will use folder names as labels.

2. create_dataset.py — extract MediaPipe hand landmarks

- Purpose: iterate through images in `./data/*`, run MediaPipe Hands in static image mode, and collect normalized hand landmark (x,y) coordinates into a feature vector per image.
- Output: `data.pickle` containing a dict with `{'data': [...], 'labels': [...]}`. The labels are the folder names (strings like `0`, `A`, etc.).
- Notes: images without detected hands are skipped.

3. train_classifier.py — train a RandomForest model

- Purpose: load `data.pickle`, split dataset (80/20), train a RandomForest classifier, evaluate accuracy and save `model.pickle`.
- How to run: `python train_classifier.py`.
- Customize: replace RandomForest with another scikit-learn model or tune hyperparameters.

4. inference_classifier.py — real-time webcam inference

- Purpose: use MediaPipe to extract landmarks from live frames, predict a class with the trained model and show bounding box + label on-screen.
- How to run: make sure `model.pickle` exists, then run `python inference_classifier.py`.
- Notes: `labels_dict` in the script maps numeric class indices to displayed characters. Modify it to match your folder names and trained labels.

Creating your own dataset (detailed)

1. Decide on class labels. You can use numbers (`0`, `1`, `2`) or textual labels (`A`, `B`, `L`). The label used is the folder name.
2. Edit `collect_imgs.py` if you want named folders. By default it uses numeric folders.
3. Run `python collect_imgs.py`. For each class the script will:
   - create `./data/<class>/`
   - show the camera stream — press `q` to start capturing for that class
   - capture `dataset_size` frames and save them as JPG files
4. When done, run `python create_dataset.py` to create `data.pickle`.
5. Train the model with `python train_classifier.py`.

Working with larger datasets

- For larger datasets, you may want to:
  - run MediaPipe with `static_image_mode=False` if you use video sequences instead of single images.
  - normalize landmark coordinates relative to hand bounding box or image size (currently code uses x,y in normalized 0..1 space provided by MediaPipe).

Tips and troubleshooting

- No camera / cv2.VideoCapture(0) fails: check that your webcam index is correct (try 0, 1, ...). Ensure no other app is locking the camera.
- MediaPipe installation errors on Windows: ensure you have a recent pip, and if necessary install Visual C++ build tools. Try `pip install mediapipe==0.10.0`.
- Skipped images in create_dataset: MediaPipe couldn't detect a hand. Try capturing clearer images, with a plain background and well-lit hand.
- Model accuracy is low: collect more varied samples, try more features (z coordinates, angles), or use a different model (SVM, MLP).

Next steps and improvements (suggested)

- Add a CLI argument parser to each script (argparse) so number of classes, dataset size, camera index, input/output file paths can be configured without editing the code.
- Save training logs and a confusion matrix.
- Replace RandomForest with a neural network (e.g., a small MLP using PyTorch or TensorFlow) if you want to improve generalization.
- Add unit tests for data-processing functions.

License & Attribution

- This repository is provided as-is for educational purposes. If you re-use the code, please credit the original author.
