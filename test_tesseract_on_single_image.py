import pytesseract
import cv2


image_path = './segmented_letters/char_12_ך.png'
# image_path = './segmented_letters/char_16_ו.png'

# custom_config = r'--oem 3 --psm 6 -l heb'
# custom_config = r'--psm 10 -l heb -c tessedit_char_whitelist=אבגדהוזחטיכלמנסעפצקרשתףץםןך'
# # Run OCR
# details = pytesseract.image_to_data(cv2.imread(image_path), config=custom_config, output_type=pytesseract.Output.DICT)
# print(details['text'])

import pytesseract
from PIL import Image

# Load test image
image = Image.open(image_path)

# For single characters
text = pytesseract.image_to_string(image, lang='heb_old', config='--psm 10')

# For text with multiple characters
# text = pytesseract.image_to_string(image, lang='heb_old', config='--psm 6')

print(text)