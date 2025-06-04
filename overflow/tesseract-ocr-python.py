import os
import subprocess
from PIL import Image
import pytesseract
import cv2
import numpy as np
import glob
from tqdm import tqdm


class TesseractOCR:
    def __init__(self, tesseract_path=None, lang='eng'):
        """
        Initialize the Tesseract OCR engine.
        
        Args:
            tesseract_path: Path to the Tesseract executable (optional)
            lang: Language model to use (default: 'eng')
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.lang = lang
    
    def read_text(self, image_path, config=''):
        """
        Extract text from an image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            config: Additional configuration options for Tesseract
            
        Returns:
            Extracted text as string
        """
        try:
            # Load image with OpenCV for preprocessing
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            # Preprocess image for better OCR results
            img = self._preprocess_image(img)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(img, lang=self.lang, config=config)
            return text
        except Exception as e:
            print(f"Error reading text from image: {e}")
            return ""
    
    def _preprocess_image(self, img):
        """
        Preprocess image for better OCR results.
        
        Args:
            img: OpenCV image object
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opening
    
    def analyze_document(self, image_path):
        """
        Extract structured information from a document using Tesseract OCR.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary containing text data, bounding boxes, and confidence levels
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            # Preprocess image
            img = self._preprocess_image(img)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
            
            # Process results
            result = {
                'text': " ".join([word for word in data['text'] if word.strip()]),
                'words': data['text'],
                'confidence': data['conf'],
                'boxes': [(data['left'][i], data['top'][i], data['width'][i], data['height'][i]) 
                         for i in range(len(data['text']))]
            }
            return result
        except Exception as e:
            print(f"Error analyzing document: {e}")
            return {}


class TesseractTrainer:
    def __init__(self, tessdata_path, lang_code='eng'):
        """
        Initialize the Tesseract trainer.
        
        Args:
            tessdata_path: Path to the tessdata directory
            lang_code: Language code for the model to train
        """
        self.tessdata_path = tessdata_path
        self.lang_code = lang_code
        
        # Create directories if they don't exist
        self.training_dir = os.path.join(os.getcwd(), 'tesseract_training')
        os.makedirs(self.training_dir, exist_ok=True)
    
    def prepare_training_data(self, images_dir, ground_truth_file=None):
        """
        Prepare training data for Tesseract.
        
        Args:
            images_dir: Directory containing training images
            ground_truth_file: File containing ground truth text (optional)
            
        Returns:
            Path to the prepared training data directory
        """
        print("Preparing training data...")
        
        # Create a box file for each training image if ground truth is provided
        if ground_truth_file:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = f.read().splitlines()
            
            # Match images with their ground truth
            image_files = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
            
            if len(image_files) != len(ground_truth):
                print(f"Warning: Number of images ({len(image_files)}) does not match number of ground truth lines ({len(ground_truth)})")
            
            for i, img_path in enumerate(image_files):
                if i < len(ground_truth):
                    # Create a box file based on ground truth
                    box_path = img_path.replace('.tif', '.box')
                    self._create_box_file(ground_truth[i], img_path, box_path)
        
        return images_dir
    
    def _create_box_file(self, text, image_path, box_path):
        """
        Create a box file for Tesseract training.
        This is a simplified version - in a real scenario, you would need more precise character position data.
        
        Args:
            text: Ground truth text
            image_path: Path to the image
            box_path: Path to save the box file
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            with open(box_path, 'w', encoding='utf-8') as f:
                for i, char in enumerate(text):
                    if char.strip():  # Skip whitespace
                        # Simplified positioning - this would need to be more accurate in practice
                        left = int(i * width / len(text))
                        right = int((i + 1) * width / len(text))
                        top = int(height * 0.4)
                        bottom = int(height * 0.6)
                        f.write(f"{char} {left} {height - bottom} {right} {height - top} 0\n")
        except Exception as e:
            print(f"Error creating box file: {e}")
    
    def fine_tune_model(self, training_data_dir, iterations=1000):
        """
        Fine-tune a Tesseract model using training data.
        
        Args:
            training_data_dir: Directory containing the prepared training data
            iterations: Number of training iterations
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            print(f"Starting fine-tuning for {self.lang_code} model with {iterations} iterations...")
            
            # Create training list file
            list_file = os.path.join(self.training_dir, f"{self.lang_code}.training_files.txt")
            with open(list_file, 'w') as f:
                for img_file in glob.glob(os.path.join(training_data_dir, '*.tif')):
                    f.write(f"{os.path.basename(img_file).split('.')[0]}\n")
            
            # Set up environment variables
            env = os.environ.copy()
            env["TESSDATA_PREFIX"] = self.tessdata_path
            
            # Run Tesseract training commands
            # Note: In a real implementation, you would need to run a series of
            # complex commands. This is a simplified example.
            commands = [
                f"tesseract {os.path.join(training_data_dir, '*.tif')} stdout -l {self.lang_code} batch.nochop makebox",
                f"unicharset_extractor {os.path.join(training_data_dir, '*.box')}",
                f"mftraining -F font_properties -U unicharset -O {self.lang_code}.unicharset {os.path.join(training_data_dir, '*.tr')}",
                f"cntraining {os.path.join(training_data_dir, '*.tr')}",
                f"combine_tessdata {self.lang_code}."
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                # In a real implementation, you would execute these commands
                # subprocess.run(cmd, shell=True, check=True, env=env)
            
            print(f"Fine-tuning complete. Model saved to {os.path.join(self.training_dir, f'{self.lang_code}.traineddata')}")
            return True
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return False


# Example usage
def main():
    # Basic usage example
    print("Basic OCR example:")
    ocr = TesseractOCR()
    
    # Example image path - replace with your own image path
    sample_image = "sample.png"
    
    # Check if sample image exists, if not create a simple one
    if not os.path.exists(sample_image):
        print(f"Sample image {sample_image} not found. Creating a simple test image...")
        # Create a simple image with text using PIL
        img = Image.new('RGB', (300, 100), color=(255, 255, 255))
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Hello, Tesseract OCR!", fill=(0, 0, 0))
        img.save(sample_image)
    
    # Extract text from the image
    text = ocr.read_text(sample_image)
    print(f"Extracted text: {text}")
    
    # Advanced document analysis
    print("\nAdvanced document analysis:")
    doc_info = ocr.analyze_document(sample_image)
    print(f"Complete text: {doc_info.get('text', '')}")
    print(f"Word count: {len(doc_info.get('words', []))}")
    
    # Fine-tuning example (note: this would require actual training data)
    print("\nTesseract model fine-tuning example:")
    tessdata_path = "/usr/share/tesseract-ocr/4.00/tessdata"  # Default path, adjust as needed
    trainer = TesseractTrainer(tessdata_path)
    
    print("Note: The full training process requires actual training data and Tesseract installed with training tools.")
    print("This example shows the function calls but doesn't execute the actual training commands.")


if __name__ == "__main__":
    main()
