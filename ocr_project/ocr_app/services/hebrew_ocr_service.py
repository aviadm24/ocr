import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
from django.conf import settings
import base64
from .func import pip_line


class HebrewOCRService:
    @staticmethod
    def process_torah_image(image_path):
        img = cv2.imread(image_path)

        ocr_text = pip_line(image_path)

        # Additional Torah-specific analysis
        torah_specific_issues = HebrewOCRService.analyze_torah_specific_issues(ocr_text, [])

        return {
            'ocr_text': ocr_text,
            # 'missing_letters': missing_letters,
            # 'confidence_data': confidence_data,
            'torah_specific_issues': torah_specific_issues,
            'processed_image_path': os.path.join('processed', os.path.basename(image_path))
        }
    @staticmethod
    def analyze_torah_specific_issues(ocr_text, confidence_data):
        """
        Analyze Torah-specific requirements for letters

        This is a simplified version and would need to be expanded with actual
        Torah scribal requirements (sofer standards)
        """
        issues = []

        # Check for commonly confused Hebrew letters in Torah scrolls
        # In a real implementation, this would have much more comprehensive checks
        confused_pairs = [
            ('ר', 'ד'),  # Resh vs Dalet
            ('ו', 'י'),  # Vav vs Yod
            ('ב', 'כ'),  # Bet vs Kaf
            ('ח', 'ת'),  # Chet vs Tav
            ('ה', 'ח')  # He vs Chet
        ]

        # Find letters with medium confidence (might be malformed but detected)
        medium_confidence_chars = [item for item in confidence_data
                                   if 60 <= item['confidence'] < 80 and len(item['text'].strip()) > 0]

        for item in medium_confidence_chars:
            text = item['text']
            for pair in confused_pairs:
                if any(char in text for char in pair):
                    issues.append({
                        'type': 'potentially_confused_letters',
                        'text': text,
                        'description': f"Potential confusion between similar letters (e.g., {'/'.join(pair)})",
                        'confidence': item['confidence']
                    })

        # Check for potentially broken letters (very low confidence single characters)
        broken_candidates = [item for item in confidence_data
                             if item['confidence'] < 60 and len(item['text'].strip()) == 1]

        for item in broken_candidates:
            issues.append({
                'type': 'potentially_broken_letter',
                'text': item['text'],
                'description': "Letter may be broken or malformed, requiring repair",
                'confidence': item['confidence']
            })

        return issues

    @staticmethod
    def preprocess_image_for_torah(image_path):
        """Special preprocessing for Torah scroll images"""
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply deskewing if the text is not perfectly horizontal
        # This is important for Torah scroll images which might be photographed at angles
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate the image to deskew it if angle is significant
        # if abs(angle) > 0.5:
        if abs(angle) > 90:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

            # Save as temporary preprocessed image
            preprocessed_path = os.path.join(settings.MEDIA_ROOT, 'preprocessed', os.path.basename(image_path))
            os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
            cv2.imwrite(preprocessed_path, rotated)

            return preprocessed_path

        return image_path  # Return original if no significant skew