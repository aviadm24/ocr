import pytesseract
import cv2
from PIL import Image


# Path to the Tesseract executable
# For Windows, it might look like this:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to detect text boxes and draw rectangles
def draw_boxes_on_text(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert image to grayscale for OCR processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to get the bounding boxes of the text
    boxes = pytesseract.image_to_boxes(gray_image, lang='heb_old', config='--psm 1')

    # Loop through each box and draw a rectangle around the detected text
    for box in boxes.splitlines():
        b = box.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

        # Tesseract's coordinates are relative to the image size; OpenCV uses (x, y) for top-left corner and (w, h) for bottom-right corner
        cv2.rectangle(image, (x, gray_image.shape[0] - y), (w, gray_image.shape[0] - h), (0, 255, 0), 2)

    # Show the image with the rectangles drawn
    cv2.imshow("Image with Text Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
image_path = 'parshiya.png'
draw_boxes_on_text(image_path)
# image = cv2.imread(image_path)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# text = pytesseract.image_to_string(gray_image, lang='heb_old', config='--psm 6')
# print(text)