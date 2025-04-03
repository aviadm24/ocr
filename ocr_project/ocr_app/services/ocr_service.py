import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
import io
import os
import json
from django.conf import settings
import base64


class OCRService:
    @staticmethod
    def process_image(image_path):
        """Process the image with OCR and detect missing letters"""
        # Read the image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Run OCR
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
        details = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

        # Create a copy of the original image for visualization
        img_with_boxes = img.copy()

        # Process OCR results to identify potential missing letters
        missing_letters = []
        confidence_data = []

        # Prepare for visualization
        for i in range(len(details['text'])):
            if int(details['conf'][i]) > 0:  # Filter out low confidence detections
                x = details['left'][i]
                y = details['top'][i]
                w = details['width'][i]
                h = details['height'][i]
                text = details['text'][i]
                conf = int(details['conf'][i])

                confidence_data.append({
                    'text': text,
                    'confidence': conf
                })

                # Draw rectangles for visualization
                if conf < 60 and len(text.strip()) > 0:  # Potential missing or incorrect letters
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for potential issues
                    missing_letters.append({
                        'text': text,
                        'position': (x, y, w, h),
                        'confidence': conf
                    })
                else:
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0),
                                  2)  # Green for confident detection

        # Save the processed image
        processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed', os.path.basename(image_path))
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        cv2.imwrite(processed_image_path, img_with_boxes)

        # Get full OCR text
        ocr_text = pytesseract.image_to_string(thresh, config=custom_config)

        return {
            'ocr_text': ocr_text,
            'missing_letters': missing_letters,
            'confidence_data': confidence_data,
            'processed_image_path': os.path.join('processed', os.path.basename(image_path))
        }