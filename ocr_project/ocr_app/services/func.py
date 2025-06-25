import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import label, binary_fill_holes
from tensorflow.keras import layers, models
from django.conf import settings
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ocr_app', 'model', 'hebrew_ocr_model.h5')


def tf_model_predict_single(model=None, char_img=None):
    def switch_colors(image):
        return cv2.bitwise_not(image)

    img = switch_colors(char_img)
    img = cv2.resize(img, (64, 64))
    roi_normalized = img.astype(np.float32) / 255.0
    roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
    pred = model.predict(roi_input)[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    letter = hebrew_chars_dict.get(pred_class, str(pred_class))
    return letter


def tf_model_predict_batch(model=None, prepared_chars=None):
    # predictions = model.predict(np.array(prepared_chars).reshape(-1, 32, 32, 1))
    desired_width = desired_height = 64
    prepared_chars = [cv2.resize(char, (desired_width, desired_height)) for char in prepared_chars]
    prepared_chars = np.array(prepared_chars).reshape(-1, desired_height, desired_width, 1)
    predictions = model.predict(prepared_chars)
    print(predictions)

    predicted_classes = np.argmax(predictions, axis=1)

    # Convert to Hebrew characters
    predicted_letters = [hebrew_chars_dict[idx] for idx in predicted_classes]

    # If you want to get confidence scores along with predictions
    confidence_scores = np.max(predictions, axis=1)

    # Create a result with both letter and confidence
    results = [(hebrew_chars_dict[idx], score) for idx, score in zip(predicted_classes, confidence_scores)]

    # Print results
    for i, (letter, confidence) in enumerate(results):
        print(f"Character {i}: {letter} (confidence: {confidence:.2f})")

    # If you want to reconstruct the text (assuming right-to-left reading)
    text = ''.join(predicted_letters[::-1])  # Reverse for right-to-left
    print("Predicted text:", text)
    return text


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return img, binary


def segment_lines_advanced(binary_image, prominence_threshold=0.1, min_line_height=10):
    """
    Advanced line segmentation algorithm with improved line detection

    Args:
        binary_image (numpy.ndarray): Binary image of the document
        prominence_threshold (float): Threshold for detecting significant line breaks
        min_line_height (int): Minimum height of a line to be considered valid

    Returns:
        tuple: List of line images and their boundaries
    """
    # Preprocess to handle skewed images
    skew_angle = 0
    # preprocessed_image, skew_angle = preprocess_image(binary_image)
    preprocessed_image = binary_image

    # Compute horizontal projection profile
    h_projection = np.sum(preprocessed_image, axis=1)

    # Normalize projection profile
    h_projection_normalized = h_projection / np.max(h_projection)

    # Find local minima (potential line breaks)
    peaks, _ = find_peaks(-h_projection_normalized,
                          height=None,
                          threshold=None,
                          prominence=prominence_threshold)

    # Sort peaks to find proper line boundaries
    peaks = sorted(peaks)

    # Extract line images
    line_images = []
    line_boundaries = []

    # If no peaks found, treat whole image as one line
    if len(peaks) == 0:
        line_images.append(preprocessed_image)
        line_boundaries.append((0, preprocessed_image.shape[0]))
        return line_images, line_boundaries

    # Add first line (from image start to first peak)
    if peaks[0] > min_line_height:
        line_images.append(preprocessed_image[:peaks[0], :])
        line_boundaries.append((0, peaks[0]))

    # Extract lines between peaks
    for i in range(len(peaks) - 1):
        line_start = peaks[i]
        line_end = peaks[i + 1]

        # Only add line if it's taller than minimum height
        if line_end - line_start >= min_line_height:
            line_images.append(preprocessed_image[line_start:line_end, :])
            line_boundaries.append((line_start, line_end))

    # Add last line (from last peak to image end)
    if preprocessed_image.shape[0] - peaks[-1] > min_line_height:
        line_images.append(preprocessed_image[peaks[-1]:, :])
        line_boundaries.append((peaks[-1], preprocessed_image.shape[0]))

    # Rotate line images back to original orientation if skew was corrected
    if skew_angle != 0:
        line_images_original = []
        line_boundaries_original = []
        for line_img, (start, end) in zip(line_images, line_boundaries):
            # Rotate back
            h, w = line_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)
            rotated_line = cv2.warpAffine(line_img, M, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)

            line_images_original.append(rotated_line)
            line_boundaries_original.append((start, end))

        return line_images_original, line_boundaries_original

    return line_images, line_boundaries


def segment_characters_advanced(line_image, min_area=50, max_area=2000):
    """
    Advanced character segmentation for Hebrew text with improved handling of complex letter shapes

    Args:
        line_image (numpy.ndarray): Binary image of a single text line
        min_area (int): Minimum area for a valid character component
        max_area (int): Maximum area for a valid character component

    Returns:
        tuple: List of character images and their positions
    """
    # Preprocessing to improve connected components detection
    # Fill holes in characters to handle broken letters
    filled_image = binary_fill_holes(line_image > 0).astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_image, connectivity=8)

    # List to store character data
    char_candidates = []

    # Collect potential character components
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Filter based on area
        if area < min_area or area > max_area:
            continue

        # Create mask for this component
        char_mask = np.zeros_like(filled_image)
        char_mask[labels == i] = 255

        char_candidates.append({
            'mask': char_mask,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'centroid': centroids[i]
        })

    # Sort candidates from right to left (for Hebrew)
    char_candidates.sort(key=lambda x: -x['x'])

    # Advanced filtering to remove nested/overlapping components
    filtered_candidates = []
    for candidate in char_candidates:
        is_nested = False

        # Check if this candidate is completely contained within another
        for other in char_candidates:
            if candidate is other:
                continue

            # Check if candidate is completely inside another component
            if (other['x'] <= candidate['x'] and
                    other['y'] <= candidate['y'] and
                    other['x'] + other['w'] >= candidate['x'] + candidate['w'] and
                    other['y'] + other['h'] >= candidate['y'] + candidate['h']):
                is_nested = True
                break

        if not is_nested:
            filtered_candidates.append(candidate)

    # Prepare final character images and positions
    char_images = []
    char_positions = []

    for candidate in filtered_candidates:
        # Add padding
        padding = 5
        x_start = max(0, candidate['x'] - padding)
        y_start = max(0, candidate['y'] - padding)
        x_end = min(line_image.shape[1], candidate['x'] + candidate['w'] + padding)
        y_end = min(line_image.shape[0], candidate['y'] + candidate['h'] + padding)

        # Extract character image
        char_img = line_image[y_start:y_end, x_start:x_end]

        # Additional filter: aspect ratio
        if 0.2 <= char_img.shape[1] / char_img.shape[0] <= 3:
            char_images.append(char_img)
            char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

    return char_images, char_positions


hebrew_chars = {
        0: 'ב', 1: 'כ', 2: 'פ', 3: 'ך', 4: 'ה', 5: 'ר', 6: 'ף', 7: 'ח', 8: 'מיקס',
        9: 'צ', 10: 'ק', 11: 'ט', 12: 'נ', 13: 'ת', 14: 'ג', 15: 'ן', 16: 'ם',
        17: 'ע', 18: 'י', 19: 'ד', 20: 'חצי-ק', 21: 'זבל', 22: 'ל', 23: 'א', 24: 'ז',
        25: 'ש', 26: 'זבל', 27: 'לא ידוע', 28: 'ו', 29: 'מ', 30: 'ץ', 31: 'ס'
    }
def pip_line(image_path):
    # Here you would load your trained model and predict on the prepared characters
    model = models.load_model(MODEL_PATH)
    model.summary()
    img, preprocessed_image = preprocess_image(image_path)
    line_images, line_boundaries = segment_lines_advanced(preprocessed_image)
    all_chars = []
    all_positions = []
    text = []
    # Process each line
    for i, line_image in enumerate(line_images):
        char_images, char_positions = segment_characters_advanced(line_image)
        
        # Store with offset for the line position
        y_offset = line_boundaries[i][0]
        adjusted_positions = [(x, y + y_offset, w, h) for x, y, w, h in char_positions]
        
        all_chars.extend(char_images)
        all_positions.extend(adjusted_positions)
        
        # Save segmented characters
        for j, char_img in enumerate(char_images):
            segmented_img = cv2.cvtColor(preprocessed_image.copy(), cv2.COLOR_GRAY2BGR)
            # x, y, w, h = all_positions[j]
            x, y, w, h = adjusted_positions[j]
            cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('letter', segmented_img)
            def switch_colors(image):
                return cv2.bitwise_not(image)
            img = switch_colors(char_img)
            img = cv2.resize(img, (64, 64))
            roi_normalized = img.astype(np.float32) / 255.0
            roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
            pred = model.predict(roi_input)[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
            letter = hebrew_chars.get(pred_class, str(pred_class))
            print(f"Character {j}: {letter} (confidence: {confidence:.2f})")
            text.append(letter)

    return ','.join(text)