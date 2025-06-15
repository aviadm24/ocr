#!/usr/bin/env python3
import os
import subprocess
import shutil
from PIL import Image
import random
import argparse


def create_training_data(letters_root_dir, output_dir, font_name="OldHebrew", lang_code="heb_old"):
    """
    Create Tesseract training data from folders of letter images.
    
    Args:
        letters_root_dir: Root directory containing one subfolder per letter
        output_dir: Directory to output training files
        font_name: Name of the font for training
        lang_code: Language code for the trained model
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    train_data_dir = os.path.join(output_dir, "traindata")
    os.makedirs(train_data_dir, exist_ok=True)
    
    # Track all generated files
    training_images = []
    box_file_content = []
    
    # Process each letter folder
    print("Processing letter folders...")
    letter_count = 0
    from heb_dict import heb_dict
    for letter_dir in sorted(os.listdir(letters_root_dir)):
        letter_path = os.path.join(letters_root_dir, letter_dir)
        print(letter_dir)
        # Skip if not a directory
        if not os.path.isdir(letter_path) or heb_dict[letter_dir] == '':
            continue
            
        # The letter is the name of the folder
        hebrew_letter = heb_dict[letter_dir]
        
        print(f"Processing letter: {hebrew_letter}")
        
        # Process each image in the letter directory
        images = [f for f in os.listdir(letter_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Use a sample of images if there are too many (optional)
        if len(images) > 50:  # Adjust sample size as needed
            images = random.sample(images, 50)
        
        for img_file in images:
            img_path = os.path.join(letter_path, img_file)
            
            # Create a unique name for this training example
            example_name = f"{hebrew_letter}_{letter_count:04d}"
            output_image_path = os.path.join(train_data_dir, f"{example_name}.tif")
            
            # Convert and save image as tif (Tesseract preferred format)
            img = Image.open(img_path)
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            img.save(output_image_path)
            
            # Get image dimensions for box file
            width, height = img.size
            
            # Create box file entry (character, left, bottom, right, top, page)
            # Note: Tesseract coordinates have origin at bottom left
            box_entry = f"{hebrew_letter} 0 0 {width} {height} 0\n"
            
            # Write to individual box file
            with open(os.path.join(train_data_dir, f"{example_name}.box"), "w", encoding="utf-8") as f:
                f.write(box_entry)
                
            training_images.append(example_name)
            letter_count += 1
    
    print(f"Created {letter_count} training examples")
    
    # Create training script
    script_path = os.path.join(output_dir, "run_training.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
cd {output_dir}

# Font name and language code
FONT_NAME="{font_name}"
LANG_CODE="{lang_code}"
TRAIN_DATA_DIR="traindata"

# Step 1: Generate training files from existing box files
for image in $TRAIN_DATA_DIR/*.tif; do
    basename=$(basename "$image" .tif)
    echo "Processing $basename"
    tesseract "$image" "$TRAIN_DATA_DIR/$basename" box.train
done

# Step 2: Extract unicharset
unicharset_extractor $TRAIN_DATA_DIR/*.box

# Step 3: Create font_properties file
echo "$FONT_NAME 0 0 0 0 0" > font_properties

# Step 4: Create clustering data
shapeclustering -F font_properties -U unicharset -O $LANG_CODE.unicharset $TRAIN_DATA_DIR/*.tr

# Step 5: Compute the character shape features
mftraining -F font_properties -U unicharset -O $LANG_CODE.unicharset $TRAIN_DATA_DIR/*.tr

# Step 6: Compute character normalization features
cntraining $TRAIN_DATA_DIR/*.tr

# Step 7: Rename files
mv inttemp $LANG_CODE.inttemp
mv normproto $LANG_CODE.normproto
mv pffmtable $LANG_CODE.pffmtable
mv shapetable $LANG_CODE.shapetable

# Step 8: Create Tesseract traineddata file
combine_tessdata $LANG_CODE.

echo "Training completed. The new language model is: $LANG_CODE.traineddata"
echo "To install system-wide, run: sudo cp $LANG_CODE.traineddata /usr/share/tesseract-ocr/4.00/tessdata/"
""")
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Training script created at {script_path}")
    print("Run this script to generate the Tesseract training data")
    print("\nIf you want to add more examples or correct the box files,")
    print(f"edit the files in {train_data_dir} before running the training script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Tesseract training data from folders of letter images")
    parser.add_argument("letters_dir", help="Root directory containing one subfolder per letter")
    parser.add_argument("output_dir", help="Directory to output training files")
    parser.add_argument("--font-name", default="OldHebrew", help="Name for the font")
    parser.add_argument("--lang-code", default="heb_old", help="Language code for the model")
    
    args = parser.parse_args()
    
    create_training_data(args.letters_dir, args.output_dir, args.font_name, args.lang_code)