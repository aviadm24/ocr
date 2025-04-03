import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import label, binary_fill_holes


def get_heb_dict():
    with open('hebrew_ocr_model.h5_classes.txt') as f:
        heb_dict = {}
        for line in f.readlines():
            line = line.split(',')
            heb_dict[int(line[0])] = line[1]
    return heb_dict


def preprocess_image(binary_image):
    """
    Preprocess the image to handle skewed text

    Args:
        binary_image (numpy.ndarray): Binary image of the document

    Returns:
        tuple: Preprocessed image and rotation angle
    """
    # Detect skew using probabilistic Hough line transform
    coords = np.column_stack(np.where(binary_image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Correct skew angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate image to correct skew
    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle


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


def post_process_segmentation(char_images, char_positions, max_overlap_ratio=0.5):
    """
    Post-process character segmentation to handle complex cases

    Args:
        char_images (list): List of character images
        char_positions (list): List of character positions
        max_overlap_ratio (float): Maximum allowed overlap ratio between characters

    Returns:
        tuple: Refined list of character images and positions
    """
    # If no characters found, return original input
    if not char_images:
        return char_images, char_positions

    # Sort positions from right to left
    sorted_indices = sorted(range(len(char_positions)),
                            key=lambda k: -char_positions[k][0])

    refined_images = []
    refined_positions = []

    for i in sorted_indices:
        current_img = char_images[i]
        current_pos = char_positions[i]

        # Check for significant overlap with previous characters
        is_valid = True
        for prev_img, prev_pos in zip(refined_images, refined_positions):
            # Calculate overlap
            x1 = max(current_pos[0], prev_pos[0])
            x2 = min(current_pos[0] + current_pos[2], prev_pos[0] + prev_pos[2])
            overlap = max(0, x2 - x1)

            # Calculate overlap ratio
            overlap_ratio = overlap / min(current_pos[2], prev_pos[2])

            if overlap_ratio > max_overlap_ratio:
                is_valid = False
                break

        if is_valid:
            refined_images.append(current_img)
            refined_positions.append(current_pos)

    return refined_images, refined_positions


def debug_segmentation(line_image, char_images, char_positions):
    """
    Visualize character segmentation for debugging

    Args:
        line_image (numpy.ndarray): Original line image
        char_images (list): List of segmented character images
        char_positions (list): List of character positions
    """
    import matplotlib.pyplot as plt

    # Visualize all characters
    plt.figure(figsize=(20, 5))
    plt.subplot(1, len(char_images) + 1, 1)
    plt.imshow(line_image, cmap='gray')
    plt.title('Original Line')
    plt.axis('off')

    for i, (char_img, pos) in enumerate(zip(char_images, char_positions), 1):
        plt.subplot(1, len(char_images) + 1, i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.title(f'Char {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Modify your existing segment_characters function to use these new methods
def new_segment_characters(line_image):
    # Advanced segmentation
    char_images, char_positions = segment_characters_advanced(line_image)

    # Post-process to refine segmentation
    # char_images, char_positions = post_process_segmentation(char_images, char_positions)

    # Optional: Debug visualization
    # debug_segmentation(line_image, char_images, char_positions)

    return char_images, char_positions