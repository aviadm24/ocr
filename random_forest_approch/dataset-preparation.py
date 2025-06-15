#!/usr/bin/env python3
"""
Hebrew OCR Dataset Preparation Script

This script helps prepare a dataset of Hebrew handwritten letters by:
1. Checking image formats and ensuring consistency
2. Splitting raw data into train/validation sets
3. Performing basic data augmentation

Usage:
    python prepare_dataset.py --input_dir raw_data --output_dir processed_data
"""

import os
import argparse
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Hebrew handwritten character dataset')
    parser.add_argument('--input_dir', required=True, help='Input directory with raw data')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data')
    parser.add_argument('--augment', action='store_true', help='Perform data augmentation')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--image_size', type=int, default=32, help='Size to normalize images to')
    return parser.parse_args()

def create_directory_structure(output_dir, split_names=None):
    """Create the output directory structure"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if split_names:
        for split in split_names:
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)
    
    return output_dir

def check_and_normalize_image(img_path, size=32):
    """Check if image is valid and normalize it"""
    try:
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return None
        
        # Resize
        img = cv2.resize(img, (size, size))
        
        # Apply thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return img
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def augment_image(img):
    """Apply data augmentation techniques to an image"""
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Rotation
    for angle in [5, -5, 10, -10]:
        h, w = img.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=0)
        augmented_images.append(rotated)
    
    # Translation
    for tx, ty in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        h, w = img.shape
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, translation_matrix, (w, h), borderValue=0)
        augmented_images.append(translated)
    
    # Add noise
    noisy = img.copy()
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    noisy = cv2.add(noisy, noise)
    augmented_images.append(noisy)
    
    # Erosion and dilation (to simulate different pen thickness)
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated = cv2.dilate(img, kernel, iterations=1)
    augmented_images.append(eroded)
    augmented_images.append(dilated)
    
    return augmented_images

def process_dataset(input_dir, output_dir, val_split=0.2, augment=False, image_size=32):
    """Process the dataset"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = create_directory_structure(output_dir / 'train')
    val_dir = create_directory_structure(output_dir / 'val')
    
    # Get all subdirectories (classes)
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {input_dir}")
    
    print(f"Found {len(class_dirs)} letter classes")
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Create output class directories
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_split))
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        print(f"Class {class_name}: {len(train_images)} training, {len(val_images)} validation images")
        
        # Process training images
        for i, img_path in enumerate(tqdm(train_images, desc=f"Training {class_name}", leave=False)):
            img = check_and_normalize_image(img_path, image_size)
            
            if img is None:
                continue
            
            # Save the normalized image
            cv2.imwrite(str(train_class_dir / f"{class_name}_{i:04d}.png"), img)
            
            # Apply augmentation if requested
            if augment:
                augmented_images = augment_image(img)
                
                for j, aug_img in enumerate(augmented_images):
                    aug_path = train_class_dir / f"{class_name}_{i:04d}_aug{j}.png"
                    cv2.imwrite(str(aug_path), aug_img)
        
        # Process validation images
        for i, img_path in enumerate(tqdm(val_images, desc=f"Validation {class_name}", leave=False)):
            img = check_and_normalize_image(img_path, image_size)
            
            if img is None:
                continue
            
            # Save the normalized image
            cv2.imwrite(str(img))