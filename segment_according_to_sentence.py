import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, label


def segment_and_verify_hebrew_text(line_image, expected_text, debug=False):
    """
    Segment Hebrew text line into letters and verify against expected text

    Args:
        line_image (numpy.ndarray): Binary image of a single text line
        expected_text (str): The exact sequence of Hebrew letters expected in the image
        debug (bool): Whether to show debug visualization

    Returns:
        tuple: List of character images, positions, and verification results
    """
    # Ensure binary image
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    # Binarize if not already binary
    if np.max(line_image) > 1:
        _, line_image = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Parameters for Hebrew text - may need adjustment based on your specific images
    min_char_width = 8  # Minimum letter width
    max_char_width = 70  # Maximum letter width
    expected_char_count = len(expected_text)

    # First attempt: Basic segmentation
    char_images, char_positions = basic_segmentation(line_image, min_char_width)

    # Check if we need to refine the segmentation
    if len(char_images) != expected_char_count:
        # Try adaptive segmentation
        char_images, char_positions = adaptive_segmentation(
            line_image,
            expected_char_count,
            min_char_width,
            max_char_width
        )

    # Generate verification results
    verification_results = verify_characters(char_images, char_positions, expected_text)

    # Optional debug visualization
    if debug:
        visualize_results(line_image, char_positions, expected_text, verification_results)

    return char_images, char_positions, verification_results


def basic_segmentation(line_image, min_char_width):
    """
    Basic segmentation based on connected components

    Args:
        line_image (numpy.ndarray): Binary image
        min_char_width (int): Minimum letter width

    Returns:
        tuple: Lists of character images and positions
    """
    # Use connected components analysis
    labeled_array, num_features = label(line_image == 255)

    # Lists to store character images and positions
    char_images = []
    char_positions = []

    # Process each connected component
    for label_id in range(1, num_features + 1):
        # Get the bounding box of this component
        mask = labeled_array == label_id
        rows, cols = np.where(mask)

        if len(rows) == 0 or len(cols) == 0:
            continue

        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)

        # Get width and height
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Skip very small components (likely noise)
        if width < min_char_width or height < min_char_width:
            continue

        # Add padding
        padding = 3
        y_start = max(0, y_min - padding)
        y_end = min(line_image.shape[0], y_max + padding + 1)
        x_start = max(0, x_min - padding)
        x_end = min(line_image.shape[1], x_max + padding + 1)

        # Extract the character image
        char_img = line_image[y_start:y_end, x_start:x_end].copy()

        char_images.append(char_img)
        char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

    # Sort characters by x-position (Hebrew reads right-to-left)
    sorted_indices = np.argsort([pos[0] for pos in char_positions])[::-1]  # Reversed for Hebrew

    sorted_char_images = [char_images[i] for i in sorted_indices]
    sorted_char_positions = [char_positions[i] for i in sorted_indices]

    return sorted_char_images, sorted_char_positions


def adaptive_segmentation(line_image, expected_count, min_char_width, max_char_width):
    """
    Adaptive segmentation trying to match expected character count

    Args:
        line_image (numpy.ndarray): Binary image
        expected_count (int): Expected number of characters
        min_char_width (int): Minimum letter width
        max_char_width (int): Maximum letter width

    Returns:
        tuple: Lists of character images and positions
    """
    # Try different erosion levels
    best_images = []
    best_positions = []
    best_diff = float('inf')

    # Try a range of erosion levels
    for erosion_size in range(0, 3):
        if erosion_size > 0:
            # Apply erosion to separate touching letters
            kernel = np.ones((1, erosion_size), np.uint8)
            eroded_image = binary_erosion(line_image == 255, structure=kernel)
            eroded_image = eroded_image.astype(np.uint8) * 255
        else:
            eroded_image = line_image.copy()

        # Segment using connected components
        char_images, char_positions = basic_segmentation(eroded_image, min_char_width)

        # Check how close we are to expected count
        diff = abs(len(char_images) - expected_count)

        if diff < best_diff:
            best_diff = diff
            best_images = char_images
            best_positions = char_positions

            # If perfect match, stop early
            if diff == 0:
                break

    # If we still have significantly fewer characters than expected, try splitting large components
    if len(best_images) < expected_count:
        # Get average letter width
        avg_width = np.mean([pos[2] for pos in best_positions])

        # Identify large components that might contain multiple letters
        new_images = []
        new_positions = []

        for i, (img, (x, y, w, h)) in enumerate(zip(best_images, best_positions)):
            if w > avg_width * 1.5 and w > min_char_width * 2:
                # Estimate number of characters in this component
                estimated_chars = min(max(2, round(w / avg_width)), expected_count - len(best_images) + 1)

                if estimated_chars > 1:
                    # Divide the component
                    segment_width = w // estimated_chars

                    for j in range(estimated_chars):
                        segment_x = x + j * segment_width
                        segment_width_actual = min(segment_width, x + w - segment_x)

                        segment_img = line_image[y:y + h, segment_x:segment_x + segment_width_actual].copy()

                        # Only add if there's enough ink
                        if np.sum(segment_img == 255) > 10:
                            new_images.append(segment_img)
                            new_positions.append((segment_x, y, segment_width_actual, h))
                else:
                    new_images.append(img)
                    new_positions.append((x, y, w, h))
            else:
                new_images.append(img)
                new_positions.append((x, y, w, h))

        # Sort again by x-position (right-to-left for Hebrew)
        sorted_indices = np.argsort([pos[0] for pos in new_positions])[::-1]  # Reversed for Hebrew

        best_images = [new_images[i] for i in sorted_indices]
        best_positions = [new_positions[i] for i in sorted_indices]

    return best_images, best_positions


def verify_characters(char_images, char_positions, expected_text):
    """
    Simple verification of characters against expected text

    Args:
        char_images (list): List of character images
        char_positions (list): List of character positions
        expected_text (str): Expected text

    Returns:
        dict: Verification results
    """
    num_expected = len(expected_text)
    num_found = len(char_images)

    # Basic verification
    verification = {
        'all_chars_found': num_found >= num_expected,
        'missing_count': max(0, num_expected - num_found),
        'extra_count': max(0, num_found - num_expected),
        'char_mapping': [],
    }

    # Map found characters to expected characters
    for i in range(min(num_found, num_expected)):
        expected_char = expected_text[i]
        verification['char_mapping'].append({
            'position': i,
            'expected_char': expected_char,
            # We don't do actual OCR here, but you could integrate it
            'is_verified': True,  # Placeholder for actual verification
        })

    return verification


def visualize_results(line_image, char_positions, expected_text, verification):
    """
    Visualize segmentation and verification results

    Args:
        line_image (numpy.ndarray): Binary image
        char_positions (list): List of character positions
        expected_text (str): Expected text
        verification (dict): Verification results
    """
    plt.figure(figsize=(20, 8))

    # Show original image with bounding boxes
    plt.subplot(2, 1, 1)
    plt.title(f'Original Image (Expected: {expected_text}, Found: {len(char_positions)} chars)')

    # Create color image for visualization
    vis_image = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)

    # Add rectangles around found characters
    for i, (x, y, w, h) in enumerate(char_positions):
        color = (0, 255, 0)  # Green by default

        # Add expected character as label
        if i < len(expected_text):
            expected_char = expected_text[i]
            label_y = y - 5 if y > 20 else y + h + 15
            cv2.putText(vis_image, expected_char,
                        (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

    # Show individual characters
    plt.subplot(2, 1, 2)

    num_chars = len(char_positions)
    if num_chars > 0:
        fig_width = min(20, num_chars * 2)
        fig = plt.gcf()
        fig.set_size_inches(fig_width, 8)

        for i, (x, y, w, h) in enumerate(char_positions):
            plt.subplot(2, num_chars, num_chars + i + 1)

            # Extract and show the character
            char_img = line_image[y:y + h, x:x + w]
            plt.imshow(char_img, cmap='gray')

            if i < len(expected_text):
                plt.title(f'{expected_text[i]}')
            else:
                plt.title(f'Extra {i}')

    plt.tight_layout()
    plt.show()


def save_segments(char_images, expected_text, output_dir='segmented_hebrew'):
    """
    Save segmented character images

    Args:
        char_images (list): List of character images
        expected_text (str): Expected text
        output_dir (str): Directory to save images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each character
    for i, img in enumerate(char_images):
        char_label = expected_text[i] if i < len(expected_text) else f'extra_{i}'
        output_path = f'{output_dir}/char_{i}_{char_label}.png'
        cv2.imwrite(output_path, img)
        print(f"Saved character {i} as {output_path}")


# Demonstration function
def verify_hebrew_text(image_path, expected_text):
    """
    Verify a line of Hebrew text against expected text

    Args:
        image_path (str): Path to the line image
        expected_text (str): Expected Hebrew text

    Returns:
        tuple: Segmentation and verification results
    """
    # Read the image
    line_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image
    _, binary_image = cv2.threshold(line_image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Segment and verify
    char_images, char_positions, verification = segment_and_verify_hebrew_text(
        binary_image, expected_text, debug=True
    )

    # Print verification results
    print("Verification Results:")
    print(f"Expected Text: {expected_text}")
    print(f"Characters Found: {len(char_images)}")
    print(f"All characters found: {verification['all_chars_found']}")

    if verification['missing_count'] > 0:
        print(f"Missing {verification['missing_count']} characters")

    if verification['extra_count'] > 0:
        print(f"Found {verification['extra_count']} extra characters")

    # Save the segmented characters
    save_segments(char_images, expected_text)

    return char_images, char_positions, verification


# Main execution
if __name__ == "__main__":
    # Example usage with Hebrew text
    # Replace with your expected Hebrew text
    expected_hebrew = "והיה כי יביאך יקוק אל ארץ הכנעני כאשר נשבע לך ולאבתיך ונתנה לך והעברת כל פטר"
    verify_hebrew_text('segmented_lines/line_1.png', expected_hebrew)