# Sign Language Detector

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
