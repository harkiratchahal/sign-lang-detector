"""
Sign Language Detector Web Application
FastAPI backend for custom sign language training and inference
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
import base64
from typing import List, Dict, Optional
from pathlib import Path
import shutil
from datetime import datetime

app = FastAPI(title="Sign Language Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = Path('./data')
MODELS_DIR = Path('./models')
STATIC_DIR = Path('./static')
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.labels_dict = {}
        self.sentence_mode = False
        self.current_sentence = []
        self.load_model()
        self.load_labels()
    
    def load_model(self):
        """Load the trained model if it exists"""
        model_path = MODELS_DIR / 'model.pickle'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_dict = pickle.load(f)
                    self.model = model_dict['model']
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print("⚠ No trained model found")
            self.model = None
    
    def load_labels(self):
        """Load label mappings"""
        labels_path = MODELS_DIR / 'labels.json'
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    self.labels_dict = json.load(f)
                print(f"✓ Loaded {len(self.labels_dict)} labels")
            except Exception as e:
                print(f"✗ Error loading labels: {e}")
                self.labels_dict = {}
        else:
            print("⚠ No labels file found")
            self.labels_dict = {}
    
    def save_labels(self):
        """Save label mappings"""
        labels_path = MODELS_DIR / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(self.labels_dict, f, indent=2)

state = AppState()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_path = STATIC_DIR / 'index.html'
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return HTMLResponse(content="<h1>Welcome to Sign Language Detector</h1><p>Frontend not found. Please check installation.</p>")

# Mount static files AFTER route definitions
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/api/status")
async def get_status():
    """Get application status"""
    return {
        "status": "online",
        "model_loaded": state.model is not None,
        "labels_count": len(state.labels_dict),
        "data_classes": len(list(DATA_DIR.iterdir())) if DATA_DIR.exists() else 0
    }

@app.get("/api/labels")
async def get_labels():
    """Get all available labels"""
    return {
        "labels": state.labels_dict,
        "count": len(state.labels_dict)
    }

@app.post("/api/labels")
async def add_label(label_name: str = Form(...), label_description: str = Form(...)):
    """Add a new label"""
    try:
        # Create directory for this label
        label_dir = DATA_DIR / label_name
        label_dir.mkdir(exist_ok=True)
        
        # Add to labels dict (use folder name as key)
        state.labels_dict[label_name] = label_description
        state.save_labels()
        
        return {
            "success": True,
            "message": f"Label '{label_name}' added successfully",
            "label": {label_name: label_description}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/labels/{label_name}")
async def delete_label(label_name: str):
    """Delete a label and its data"""
    try:
        # Remove directory
        label_dir = DATA_DIR / label_name
        if label_dir.exists():
            shutil.rmtree(label_dir)
        
        # Remove from labels dict
        if label_name in state.labels_dict:
            del state.labels_dict[label_name]
            state.save_labels()
        
        return {
            "success": True,
            "message": f"Label '{label_name}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/capture")
async def capture_images(
    label_name: str = Form(...),
    num_images: int = Form(100),
    frame_data: str = Form(...)
):
    """Capture and save images for a label"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Create directory
        label_dir = DATA_DIR / label_name
        label_dir.mkdir(exist_ok=True)
        
        # Count existing images
        existing_images = len(list(label_dir.glob('*.jpg')))
        
        # Save image
        img_path = label_dir / f'{existing_images}.jpg'
        cv2.imwrite(str(img_path), frame)
        
        return {
            "success": True,
            "count": existing_images + 1,
            "total_needed": num_images,
            "message": f"Captured {existing_images + 1}/{num_images}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_model():
    """Train the model on collected data"""
    try:
        # Import training functions
        from train_classifier import train_classifier_from_data
        from create_dataset import create_dataset
        
        # Step 1: Create dataset
        print("Creating dataset from images...")
        data_file = create_dataset()
        
        if not data_file:
            raise Exception("Failed to create dataset")
        
        # Step 2: Train model
        print("Training model...")
        model_path, accuracy = train_classifier_from_data(data_file)
        
        # Reload model
        state.load_model()
        
        return {
            "success": True,
            "accuracy": float(accuracy),
            "message": f"Model trained successfully with {accuracy*100:.2f}% accuracy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    """WebSocket endpoint for real-time inference"""
    await websocket.accept()
    
    if state.model is None:
        await websocket.send_json({
            "error": "No trained model available. Please train a model first."
        })
        await websocket.close()
        return
    
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 image
            img_data = base64.b64decode(frame_data['frame'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(frame_rgb)
            
            response = {
                "prediction": None,
                "confidence": 0.0,
                "landmarks": []
            }
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    # Extract landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)
                    
                    # Make prediction
                    try:
                        prediction = state.model.predict([np.asarray(data_aux)])
                        probabilities = state.model.predict_proba([np.asarray(data_aux)])[0]
                        confidence = float(max(probabilities))
                        
                        predicted_label = str(prediction[0])
                        predicted_text = state.labels_dict.get(predicted_label, predicted_label)
                        
                        # Calculate bounding box
                        x1 = int(min(x_) * W)
                        y1 = int(min(y_) * H)
                        x2 = int(max(x_) * W)
                        y2 = int(max(y_) * H)
                        
                        response = {
                            "prediction": predicted_text,
                            "confidence": confidence,
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            }
                        }
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        hands.close()

@app.get("/api/data/stats")
async def get_data_stats():
    """Get statistics about collected data"""
    try:
        stats = {}
        total_images = 0
        
        if DATA_DIR.exists():
            for label_dir in DATA_DIR.iterdir():
                if label_dir.is_dir():
                    count = len(list(label_dir.glob('*.jpg')))
                    stats[label_dir.name] = count
                    total_images += count
        
        return {
            "stats": stats,
            "total_images": total_images,
            "num_classes": len(stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentence/add")
async def add_to_sentence(sign: str = Form(...)):
    """Add a sign to the current sentence"""
    state.current_sentence.append(sign)
    return {
        "success": True,
        "sentence": state.current_sentence,
        "text": " ".join(state.current_sentence)
    }

@app.post("/api/sentence/clear")
async def clear_sentence():
    """Clear the current sentence"""
    state.current_sentence = []
    return {
        "success": True,
        "message": "Sentence cleared"
    }

@app.get("/api/sentence")
async def get_sentence():
    """Get the current sentence"""
    return {
        "sentence": state.current_sentence,
        "text": " ".join(state.current_sentence)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
