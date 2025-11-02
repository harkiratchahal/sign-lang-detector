import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os

def train_classifier_from_data(data_file='./models/data.pickle', output_model='./models/model.pickle'):
    """
    Train a RandomForest classifier on the extracted hand landmark data.
    
    Args:
        data_file: Path to the pickle file containing data and labels
        output_model: Path to save the trained model
    
    Returns:
        Tuple of (model_path, accuracy) or (None, 0) if failed
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print("Please run create_dataset.py first to generate the dataset.")
        return None, 0
    
    print(f"Loading data from: {data_file}")
    
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.array(data_dict['data'])
    labels = np.array(data_dict['labels'])
    
    print(f"Dataset info:")
    print(f"  - Total samples: {len(data)}")
    print(f"  - Features per sample: {len(data[0])}")
    print(f"  - Unique classes: {len(set(labels))}")
    print(f"  - Classes: {sorted(set(labels))}")
    
    # Check if all samples have the same feature length
    feature_lengths = [len(d) for d in data]
    if len(set(feature_lengths)) > 1:
        print("Warning: Not all samples have the same feature length!")
        print(f"Feature lengths found: {set(feature_lengths)}")
        # Filter to keep only samples with the most common feature length
        most_common_length = max(set(feature_lengths), key=feature_lengths.count)
        mask = [len(d) == most_common_length for d in data]
        data = data[mask]
        labels = labels[mask]
        print(f"Filtered to {len(data)} samples with {most_common_length} features")

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )
    
    print(f"\nTraining set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Train model
    print("\nTraining RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    
    # Evaluate
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    
    print(f"\n✓ Training complete!")
    print(f"  Accuracy: {score * 100:.2f}%")
    
    # Show per-class accuracy
    unique_labels = sorted(set(labels))
    print(f"\nPer-class accuracy:")
    for label in unique_labels:
        mask = y_test == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_predict[mask])
            print(f"  {label}: {class_acc * 100:.2f}%")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    # Save model
    with open(output_model, 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f"\n✓ Model saved to: {output_model}")
    
    return output_model, score

if __name__ == "__main__":
    # Run as standalone script
    train_classifier_from_data()
