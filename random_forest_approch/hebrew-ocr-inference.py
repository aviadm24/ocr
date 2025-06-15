#!/usr/bin/env python3
"""
Hebrew OCR Inference Script

This script uses a trained Random Forest model to recognize Hebrew handwritten letters
from input images.

Usage:
    python infer_hebrew_ocr.py --model path/to/model.pkl --image path/to/image.jpg
    python infer_hebrew_ocr.py --model path/to/model.pkl --dir path/to/image/directory
"""

import os
import argparse
import numpy as np
import cv2
import joblib
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Use trained model for Hebrew OCR')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--image', help='Path to a single image for recognition')
    parser.add_argument('--dir', help='Directory containing images for batch recognition')
    parser.add_argument('--output', help='Path to save the output results')
    parser.add_argument('--visualize', action='store_true', help='Visualize the recognition results')
    return parser.parse_args()


def extract_hog_features(img, image_size):
    """Extract HOG features from an image"""
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

    def switch_colors(image):
        return cv2.bitwise_not(image)
    img = switch_colors(img)
    # Compute HOG features
    hog_features = hog.compute(img)
    
    return hog_features.flatten()


def load_and_preprocess_image(image_path, image_size, use_hog=True):
    """Load, preprocess and extract features from an image"""
    # Read image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    
    if use_hog:
        features = extract_hog_features(img, image_size)
    else:
        # Resize to fixed size
        img = cv2.resize(img, (image_size, image_size))
        
        # Apply thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Normalize pixel values
        img = img / 255.0
        
        # Flatten the image to a 1D array (feature vector)
        features = img.flatten()
    
    return features, img


def recognize_image(model, image_path, image_size, use_hog=True):
    """Recognize a single image"""
    # Load and preprocess the image
    features, original_img = load_and_preprocess_image(image_path, image_size, use_hog)
    
    # Reshape features to match model input
    features = features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [(model.classes_[i], probabilities[i]) for i in top_indices]
    
    return prediction, top_predictions, original_img

def process_single_image(model_data, image_path, visualize=False, output_path=None):
    """Process a single image and print the result"""
    model = model_data['model']
    image_size = model_data['image_size']
    use_hog = model_data.get('use_hog', True)
    
    # Recognize the image
    prediction, top_predictions, original_img = recognize_image(model, image_path, image_size, use_hog)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"Recognized as: {prediction}")
    print("Top 3 predictions:")
    for letter, prob in top_predictions:
        print(f"  {letter}: {prob:.4f}")
    
    # Visualize if requested
    if visualize:
        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f"Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar([p[0] for p in top_predictions], [p[1] for p in top_predictions])
        plt.title("Top Predictions")
        plt.ylim(0, 1)
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    return prediction, top_predictions

def process_directory(model_data, directory, output_dir=None, visualize=False):
    """Process all images in a directory"""
    dir_path = Path(directory)
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpeg'))
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    results = {}
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Output path for visualization
            if output_dir:
                viz_output = output_path / f"{img_path.stem}_result.png"
            else:
                viz_output = None
            
            # Process the image
            prediction, top_predictions = process_single_image(
                model_data, 
                img_path, 
                visualize=visualize,
                output_path=viz_output
            )
            
            # Store the result
            results[str(img_path)] = {
                'prediction': prediction,
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results to a file if output directory is specified
    if output_dir:
        result_file = output_path / "recognition_results.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            for img_path, result in results.items():
                f.write(f"Image: {img_path}\n")
                f.write(f"Predicted: {result['prediction']}\n")
                f.write("Top predictions:\n")
                for letter, prob in result['top_predictions']:
                    f.write(f"  {letter}: {prob:.4f}\n")
                f.write("\n")
        
        print(f"Recognition results saved to {result_file}")
    
    return results

def segment_and_recognize_text(model_data, image_path, output_path=None, visualize=False):
    """Segment a document image into individual characters and recognize them"""
    # Load the image
    original_img = cv2.imread(str(image_path))
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from right to left (for Hebrew)
    # Get bounding rectangles for each contour
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort by x-coordinate (in reverse for Hebrew right-to-left)
    bounding_boxes.sort(key=lambda x: x[0], reverse=True)
    
    # Extract and recognize each character
    recognized_text = ""
    character_images = []
    
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Skip very small contours (noise)
        if w < 10 or h < 10:
            continue
        
        # Extract the character
        char_img = gray_img[y:y+h, x:x+w]
        
        # Add some padding
        padding = 5
        padded_img = cv2.copyMakeBorder(char_img, padding, padding, padding, padding, 
                                        cv2.BORDER_CONSTANT, value=255)
        
        # Store the character image
        character_images.append((padded_img, (x, y, w, h)))
        
        # Preprocess and recognize
        try:
            features = extract_hog_features(padded_img, model_data['image_size'])
            features = features.reshape(1, -1)
            
            # Make prediction
            prediction = model_data['model'].predict(features)[0]
            recognized_text += prediction
            
            # Draw rectangle around the character
            cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(original_img, prediction, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"Error recognizing character: {e}")
    
    # Reverse the text to get correct right-to-left Hebrew
    # recognized_text = recognized_text[::-1]
    
    print(f"Recognized text: {recognized_text}")
    
    # Save or display the visualization
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("OCR Result")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path)
            print(f"OCR visualization saved to {output_path}")
        else:
            plt.show()
    
    return recognized_text

def main():
    # Parse command line arguments
    # args = parse_args()
    class args:
        model = 'hebrew_ocr_model.pkl'
        # image = '/Users/aviadmoshe/PycharmProjects/ocr/parshiya.png'
        image = '/Users/aviadmoshe/PycharmProjects/ocr/segmented_letters/char_14_י.png'
        output = 'result.png'
        visualize = True

    # Load model
    print(f"Loading model from {args.model}...")
    model_data = joblib.load(args.model)
    print(f"Model loaded. Recognizes {len(model_data['class_names'])} Hebrew letters.")
    
    # Process input
    if args.image:
        image_path = Path(args.image)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                output_path = output_path / f"{image_path.stem}_result.png"
        else:
            output_path = None
        
        # Check if image is likely a document with multiple characters
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
            # This might be a document with multiple characters
            print("Detected a document image. Performing text segmentation and recognition...")
            segment_and_recognize_text(model_data, image_path, output_path, args.visualize)
        else:
            # This is likely a single character
            process_single_image(model_data, image_path, args.visualize, output_path)
    
    elif args.dir:
        process_directory(model_data, args.dir, args.output, args.visualize)
    
    else:
        print("Error: Either --image or --dir must be specified.")
        return

if __name__ == "__main__":
    main()
