import easyocr
import cv2
import numpy as np

# Initialize the reader for Hebrew language
reader = easyocr.Reader(['he'])

# Define the list of Hebrew letters to recognize
hebrew_letters = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר',
                  'ש', 'ת']


# Function to recognize a single letter
def recognize_single_hebrew_letter(image_path):
    # Read the image using OpenCV (you can replace this with a different method if needed)
    img = cv2.imread(image_path)

    # Perform OCR
    results = reader.readtext(img, detail=1, paragraph=False)

    # Extract recognized text and check if it matches our allowed list
    recognized_text = [result[1] for result in results]

    # Filter results to only keep Hebrew letters in our list
    filtered_results = [text for text in recognized_text if text in hebrew_letters]

    return filtered_results

image_path = '../segmented_letters/char_12_ך.png'
# image_path = './segmented_letters/char_16_ו.png'
# Test the function
recognized_letters = recognize_single_hebrew_letter(image_path)

print("Recognized Letters:", recognized_letters)
