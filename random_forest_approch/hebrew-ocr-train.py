#!/usr/bin/env python3
"""
Hebrew OCR Training Script

This script trains a Random Forest model for Hebrew handwritten letter recognition.
It expects a directory structure where each subdirectory is named after a Hebrew letter
and contains images of that letter.

Usage:
    python train_hebrew_ocr.py --data_dir path/to/dataset --output_model model.pkl
"""

import os
import argparse
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Random Forest model for Hebrew OCR')
    parser.add_argument('--data_dir', required=True, help='Directory containing image folders')
    parser.add_argument('--output_model', default='hebrew_ocr_model.pkl', help='Output model file')
    parser.add_argument('--image_size', default=32, type=int, help='Size to resize images to')
    parser.add_argument('--test_size', default=0.2, type=float, help='Proportion of data for testing')
    parser.add_argument('--n_estimators', default=100, type=int, help='Number of trees in the forest')
    parser.add_argument('--max_depth', default=None, type=int, help='Max depth of trees')
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs to run in parallel')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level')
    return parser.parse_args()

def load_and_preprocess_image(image_path, image_size):
    """Load, preprocess and extract features from an image"""
    # Read image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Resize to fixed size
    img = cv2.resize(img, (image_size, image_size))
    
    # Apply thresholding to handle different lighting conditions
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Flatten the image to a 1D array (feature vector)
    features = img.flatten()
    
    return features

def extract_hog_features(image_path, image_size):
    """Extract HOG (Histogram of Oriented Gradients) features from an image"""
    # Read image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Resize to fixed size
    img = cv2.resize(img, (image_size, image_size))
    
    # Apply thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # HOG parameters
    win_size = (image_size, image_size)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    
    # Compute HOG features
    hog_features = hog.compute(img)
    
    return hog_features.flatten()

def load_dataset(data_dir, image_size, use_hog=True):
    """Load the dataset from the directory structure"""
    data_dir = Path(data_dir)
    features = []
    labels = []
    class_names = []
    
    # Check if the data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Get a list of subdirectories (each subdirectory is a class/letter)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {data_dir}")
    
    print(f"Found {len(subdirs)} classes")
    
    # Process each class
    for class_dir in tqdm(subdirs, desc="Processing classes"):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Get all image files in the class directory
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Processing {len(image_files)} images for class {class_name}")
        
        # Process each image in the class
        for img_path in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
            if use_hog:
                feature_vector = extract_hog_features(img_path, image_size)
            else:
                feature_vector = load_and_preprocess_image(img_path, image_size)
                
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(class_name)
    
    return np.array(features), np.array(labels), class_names

def train_model(features, labels, n_estimators=100, max_depth=None, n_jobs=-1, verbose=1):
    """Train a Random Forest classifier"""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42
    )
    
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

# def visualize_predictions(X_test, y_test, y_pred, class_names, image_size, n_samples=5):
#     """Visualize some predictions"""
#     # Create a figure with subplots
#     fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3))
#
#     # Get random indices
#     indices = np.random.choice(len(X_test), n_samples, replace=False)
#
#     for i, idx in enumerate(indices):
#         # Reshape features back to image
#         img = X_test[idx].reshape(image_size, image_size)
#
#         # Plot original image
#         axes[i, 0].imshow(img, cmap='gray')
#         axes[i, 0].set_title(f"True: {y_test[idx]}")
#         axes[i, 0].axis('off')
#
#         # Plot heatmap of feature importance
#         if hasattr(model, 'feature_importances_'):
#             # Reshape feature importances to image size
#             feature_importance = model.feature_importances_[:image_size*image_size]
#             feature_importance = feature_importance.reshape(image_size, image_size)
#
#             im = axes[i, 1].imshow(feature_importance, cmap='hot')
#             axes[i, 1].set_title(f"Predicted: {y_pred[idx]}")
#             axes[i, 1].axis('off')
#
#     plt.tight_layout()
#     plt.savefig('prediction_visualization.png')
#     plt.close()


def visualize_predictions(X_test, y_test, y_pred, class_names, image_size, n_samples=5):
    """Visualize some predictions"""
    # Create a figure with subplots
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3))

    # Get random indices
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(indices):
        # Check if we're using HOG features or raw pixels
        if X_test[idx].size != image_size * image_size:
            # For HOG features, we can't reshape to the original image
            # Instead, create a blank image to represent the sample
            img = np.zeros((image_size, image_size))
            axes[i, 0].text(image_size / 4, image_size / 2, f"HOG Features\nCan't visualize directly",
                            fontsize=9, ha='center')
        else:
            # For raw pixel features, reshape to the original image dimensions
            img = X_test[idx].reshape(image_size, image_size)

        # Plot original image or placeholder
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"True: {y_test[idx]}")
        axes[i, 0].axis('off')

        # Plot prediction confidence
        prediction_probs = model.predict_proba([X_test[idx]])[0]
        class_idx = np.where(model.classes_ == y_pred[idx])[0][0]
        confidence = prediction_probs[class_idx]

        axes[i, 1].bar(['Confidence'], [confidence])
        axes[i, 1].set_title(f"Predicted: {y_pred[idx]}")
        axes[i, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()


def save_model(model, output_path, class_names):
    """Save the trained model and class names"""
    # Create a dictionary with the model and class names
    model_data = {
        'model': model,
        'class_names': class_names,
        'image_size': args.image_size,
        'use_hog': True  # We're using HOG features
    }
    
    # Save the model data
    joblib.dump(model_data, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Parse command line arguments
    # args = parse_args()
    class args:
        data_dir = '/Users/aviadmoshe/PycharmProjects/ocr/hebrew_letters_dataset'
        image_size = 64
        test_size = 0.2
        n_estimators = 100
        max_depth = None
        n_jobs = 1
        verbose = 1
        output_model = 'hebrew_ocr_model.pkl'
    # Load and preprocess the dataset
    print("Loading dataset...")
    features, labels, class_names = load_dataset(args.data_dir, args.image_size)
    # features, labels, class_names = load_dataset('/Users/aviadmoshe/PycharmProjects/ocr/hebrew_letters_dataset',
    #                                              64)
    
    # Train the model
    model, X_test, y_test = train_model(
        features, labels, 
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )
    
    # Get predictions for visualization
    y_pred = model.predict(X_test)
    
    # Visualize some predictions
    visualize_predictions(X_test, y_test, y_pred, class_names, args.image_size)
    
    # Save the model
    save_model(model, args.output_model, class_names)
    
    print("Training completed successfully!")
