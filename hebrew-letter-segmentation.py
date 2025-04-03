import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from utils import segment_lines_advanced, get_heb_dict, new_segment_characters
# Create directories for output
os.makedirs('segmented_letters', exist_ok=True)
hebrew_chars = {
        0: 'ב', 1: 'כ', 2: 'פ', 3: 'ך', 4: 'ה', 5: 'ר', 6: 'ף', 7: 'ח', 8: 'מיקס',
        9: 'צ', 10: 'ק', 11: 'ט', 12: 'נ', 13: 'ת', 14: 'ג', 15: 'ן', 16: 'ם',
        17: 'ע', 18: 'י', 19: 'ד', 20: 'חצי-ק', 21: 'זבל', 22: 'ל', 23: 'א', 24: 'ז',
        25: 'ש', 26: 'זבל', 27: 'לא ידוע', 28: 'ו', 29: 'מ', 30: 'ץ', 31: 'ס'
    }
heb_dict = get_heb_dict()


# 1. Load and preprocess the image
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


# 2. Line segmentation
def segment_lines(binary_image):
    # Find horizontal projection profile
    h_projection = np.sum(binary_image, axis=1)
    
    # Find line boundaries
    line_boundaries = []
    in_line = False
    start = 0
    
    for i, count in enumerate(h_projection):
        if count > 0 and not in_line:
            in_line = True
            start = i
        elif count == 0 and in_line:
            in_line = False
            line_boundaries.append((start, i))
    
    # Handle case where the last line extends to the bottom of the image
    if in_line:
        line_boundaries.append((start, len(h_projection)))
    
    # Extract line images
    line_images = []
    for start, end in line_boundaries:
        line_images.append(binary_image[start:end, :])
    
    return line_images, line_boundaries


# 3. Character segmentation using connected components
def segment_characters(line_image):
    # Find connected components (letters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_image, connectivity=8)
    
    # Filter out very small components (noise)
    min_area = 50  # Adjust based on your image resolution
    
    char_images = []
    char_positions = []
    
    # Start from 1 to skip the background (which is labeled as 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < min_area:
            continue
        
        # Extract the character
        char_mask = np.zeros_like(line_image)
        char_mask[labels == i] = 255
        
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(char_mask)
        
        # Add padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(line_image.shape[1], x + w + padding)
        y_end = min(line_image.shape[0], y + h + padding)
        
        char_img = line_image[y_start:y_end, x_start:x_end]
        
        char_images.append(char_img)
        char_positions.append((x, y, w, h))
    
    # Sort characters from right to left (for Hebrew)
    char_data = sorted(zip(char_images, char_positions), key=lambda x: -x[1][0])
    
    return [img for img, pos in char_data], [pos for img, pos in char_data]


# 4. Prepare segmented characters for recognition
def prepare_for_recognition(char_image, target_size=(32, 32)):
    # Resize to fixed dimensions
    resized = cv2.resize(char_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Ensure binary values
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    # Normalize pixel values to [0, 1]
    normalized = binary / 255.0
    
    return normalized


# 5. Create a simple CNN model for Hebrew character recognition
def create_hebrew_ocr_model(num_classes=27):  # 22 Hebrew letters + 5 final forms
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Main process
def segment_hebrew_text(image_path, visualize=True):
    model = models.load_model('hebrew_ocr_model.h5')
    # Load and preprocess
    original, binary = preprocess_image(image_path)
    cv2.imshow('parsha', original)
    # Segment lines
    line_images, line_boundaries = segment_lines_advanced(binary)
    
    all_chars = []
    all_positions = []
    
    # Process each line
    for i, line_image in enumerate(line_images):
        # line_save_path = os.path.join('segmented_lines', f'line_{i}.png')
        # cv2.imwrite(line_save_path, line_image)
        # Segment characters in the line
        char_images, char_positions = new_segment_characters(line_image)
        
        # Store with offset for the line position
        y_offset = line_boundaries[i][0]
        adjusted_positions = [(x, y + y_offset, w, h) for x, y, w, h in char_positions]
        
        all_chars.extend(char_images)
        all_positions.extend(adjusted_positions)
        
        # Save segmented characters
        for j, char_img in enumerate(char_images):
            segmented_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            # x, y, w, h = all_positions[j]
            x, y, w, h = adjusted_positions[j]
            cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('letter', segmented_img)
            def switch_colors(image):
                return cv2.bitwise_not(image)
            img = switch_colors(char_img)
            img = cv2.resize(img, (64, 64))
            roi_normalized = img.astype(np.float32) / 255.0
            roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
            pred = model.predict(roi_input)[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
            letter = heb_dict.get(pred_class, str(pred_class))
            cv2.imshow(letter, char_img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            # save_path = os.path.join('segmented_letters', f'line_{i}_char_{j}.png')
            # cv2.imwrite(save_path, char_img)
    cv2.destroyAllWindows()
    # Prepare all characters for recognition
    prepared_chars = [prepare_for_recognition(char) for char in all_chars]
    
    # Visualize if requested
    if visualize:
        # Plot original image
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        # Plot binary image with segmentation boxes
        plt.subplot(2, 1, 2)
        segmented_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
        
        # Draw rectangles for each character
        for x, y, w, h in all_positions:
            cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
        plt.title('Segmented Characters')
        plt.tight_layout()
        plt.savefig('segmentation_result.png')
        plt.show()
    
    return prepared_chars, all_positions


def switch_colors(image):
    # Invert the colors of the image
    return cv2.bitwise_not(image)


# Run the segmentation
if __name__ == "__main__":
    image_path = "parshiya.png"  # Replace with your image path
    prepared_chars, positions = segment_hebrew_text(image_path, visualize=False)
    print(f"Segmented {len(prepared_chars)} Hebrew characters")
    
    # Here you would load your trained model and predict on the prepared characters
    model = models.load_model('hebrew_ocr_model.h5')
    model.summary()
    # predictions = model.predict(np.array(prepared_chars).reshape(-1, 32, 32, 1))
    desired_width = desired_height = 64
    prepared_chars = [cv2.resize(char, (desired_width, desired_height)) for char in prepared_chars]
    prepared_chars = np.array(prepared_chars).reshape(-1, desired_height, desired_width, 1)
    predictions = model.predict(prepared_chars)
    print(predictions)

    predicted_classes = np.argmax(predictions, axis=1)

    # Convert to Hebrew characters
    predicted_letters = [hebrew_chars[idx] for idx in predicted_classes]

    # If you want to get confidence scores along with predictions
    confidence_scores = np.max(predictions, axis=1)

    # Create a result with both letter and confidence
    results = [(hebrew_chars[idx], score) for idx, score in zip(predicted_classes, confidence_scores)]

    # Print results
    for i, (letter, confidence) in enumerate(results):
        print(f"Character {i}: {letter} (confidence: {confidence:.2f})")

    # If you want to reconstruct the text (assuming right-to-left reading)
    text = ''.join(predicted_letters[::-1])  # Reverse for right-to-left
    print("Predicted text:", text)

    plt.figure(figsize=(15, 10))
    for i, (char_img, (letter, conf)) in enumerate(zip(prepared_chars[:20], results[:20])):
        plt.subplot(4, 5, i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.title(f"{letter} ({conf:.2f})")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()
