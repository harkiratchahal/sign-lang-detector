import mediapipe as mp
import cv2
import os
import pickle

def create_dataset(data_dir='./data', output_file='./models/data.pickle'):
    """
    Extract MediaPipe hand landmarks from images and save to pickle file.
    Supports custom text labels (folder names become labels).
    
    Args:
        data_dir: Directory containing image folders (each folder is a class)
        output_file: Path to save the pickle file
    
    Returns:
        Path to the created pickle file or None if failed
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []
    
    print(f"Processing images from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return None
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_processed = 0
    skipped = 0
    
    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(dir_path):
            continue
            
        print(f"Processing class: {dir_}")
        class_count = 0
        
        for img_path in os.listdir(dir_path):
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            data_aux = []
            img = cv2.imread(os.path.join(dir_path, img_path))
            
            if img is None:
                skipped += 1
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)
                class_count += 1
                total_processed += 1
            else:
                skipped += 1
        
        print(f"  ✓ Processed {class_count} images for class '{dir_}'")
    
    if len(data) == 0:
        print("Error: No hand landmarks detected in any images!")
        return None
    
    print(f"\nTotal processed: {total_processed}, Skipped: {skipped}")
    
    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"✓ Dataset saved to: {output_file}")
    hands.close()
    return output_file

if __name__ == "__main__":
    # Run as standalone script
    create_dataset()
