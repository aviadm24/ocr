import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2
import os

hebrew_chars = {
        0: 'ב', 1: 'כ', 2: 'פ', 3: 'ך', 4: 'ה', 5: 'ר', 6: 'ף', 7: 'ח', 8: 'מיקס',
        9: 'צ', 10: 'ק', 11: 'ט', 12: 'נ', 13: 'ת', 14: 'ג', 15: 'ן', 16: 'ם',
        17: 'ע', 18: 'י', 19: 'ד', 20: 'חצי-ק', 21: 'זבל', 22: 'ל', 23: 'א', 24: 'ז',
        25: 'ש', 26: 'זבל', 27: 'לא ידוע', 28: 'ו', 29: 'מ', 30: 'ץ', 31: 'ס'
    }
img_height = 64
img_width = 64
images = []
results = []
model = models.load_model('hebrew_ocr_model.h5')
for image_path in os.listdir('segmented_letters'):
    try:
        # Load image and convert to grayscale
        full_path = os.path.join(os.getcwd(),'segmented_letters', image_path)
        img = cv2.imread(full_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        def switch_colors(image):
            # Invert the colors of the image
            return cv2.bitwise_not(image)
        img = switch_colors(img)
        img = cv2.resize(img, (img_width, img_height))

        # Normalize pixel values
        roi_normalized = img.astype(np.float32) / 255.0

        # Add channel dimension (TensorFlow expects this)
        # roi_input = np.expand_dims(img, axis=-1)
        roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)

        # Make prediction

        pred = model.predict(roi_input)[0]
        pred_class = np.argmax(pred)
        confidence = pred[pred_class]

        # Get Hebrew letter
        letter = hebrew_chars.get(pred_class, str(pred_class))
        images.append(img)
        results.append((letter,confidence))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
plt.figure(figsize=(15, 10))
for i, (char_img, (letter, conf)) in enumerate(zip(images[:20], results[:20])):
    plt.subplot(4, 5, i + 1)
    plt.imshow(char_img, cmap='gray')
    plt.title(f"{letter} ({conf:.2f})")
    plt.axis('off')
plt.tight_layout()
plt.savefig('prediction_results.png')
plt.show()