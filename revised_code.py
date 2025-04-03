import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, label
from difflib import SequenceMatcher


def find_letter_separation(line_image, start_x, end_x):
    """
    Basic letter separation check for initial segmentation

    Args:
        line_image (numpy.ndarray): Binary image of the line
        start_x (int): Start x-coordinate of potential letter
        end_x (int): End x-coordinate of potential letter

    Returns:
        bool: True if a potential separation is found, False otherwise
    """
    separation_region = line_image[:, start_x:end_x]

    # Compute vertical projection
    v_projection = np.sum(separation_region == 255, axis=0)

    # Check for potential separation
    low_ink_threshold = separation_region.shape[0] * 0.1  # 10% of column height
    potential_separations = np.where(v_projection < low_ink_threshold)[0]

    # If we find potential separation points
    if len(potential_separations) > 0:
        return True

    return False


def attempt_advanced_separation(line_image, start_x, end_x):
    """
    More advanced letter separation check for difficult cases

    Args:
        line_image (numpy.ndarray): Binary image of the line
        start_x (int): Start x-coordinate of region
        end_x (int): End x-coordinate of region

    Returns:
        list: List of x-coordinates for potential separation points
    """
    region = line_image[:, start_x:end_x]
    region_height = region.shape[0]
    region_width = region.shape[1]

    # If region is too narrow, can't separate further
    if region_width < 15:  # Adjust based on your letter sizes
        return []

    separation_points = []

    # 1. Check with targeted erosion
    h_kernel = np.ones((1, 2), np.uint8)  # Horizontal kernel to break vertical connections
    eroded = binary_erosion(region == 255, structure=h_kernel, iterations=1)

    # Check connected components after erosion
    labeled_array, num_features = label(eroded)

    if num_features > 1:
        # Find boundaries between components
        for i in range(1, region_width):
            left_components = set(labeled_array[:, i - 1].flatten())
            right_components = set(labeled_array[:, i].flatten())

            # Remove background (0)
            if 0 in left_components: left_components.remove(0)
            if 0 in right_components: right_components.remove(0)

            # If components differ, this might be a separation point
            if left_components and right_components and not left_components.intersection(right_components):
                separation_points.append(start_x + i)

    # 2. Check ink density
    if not separation_points:
        v_projection = np.sum(region == 255, axis=0)

        # Find local minima in projection
        for i in range(1, region_width - 1):
            if (v_projection[i] < v_projection[i - 1] and
                    v_projection[i] < v_projection[i + 1] and
                    v_projection[i] < np.mean(v_projection) * 0.5):
                separation_points.append(start_x + i)

    return separation_points


def segment_with_expected_text(line_image, expected_text, debug=False):
    """
    Character segmentation with knowledge of expected text

    Args:
        line_image (numpy.ndarray): Binary image of a single text line
        expected_text (str): The text that should be in the image
        debug (bool): Whether to show debug visualization

    Returns:
        tuple: List of character images and their positions
    """
    # Ensure binary image
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    # Binarize if not already binary
    if np.max(line_image) > 1:
        _, line_image = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute vertical projection (ink density)
    v_projection = np.sum(line_image == 255, axis=0)

    # Parameters
    min_char_width = 10  # Minimum character width
    max_char_width = 50  # Maximum width for typical characters

    # Lists to store character images and positions
    char_images = []
    char_positions = []

    # Basic sliding window approach (similar to your original code)
    current_start = None
    in_char = False

    for x in range(line_image.shape[1]):
        # Check ink density in vertical slice
        slice_ink_density = np.sum(line_image[:, x] == 255)

        # Detect character start
        if slice_ink_density > 0 and not in_char:
            current_start = x
            in_char = True

        # Detect character end
        elif slice_ink_density == 0 and in_char:
            char_end = x

            # Check if character meets width criteria
            if char_end - current_start >= min_char_width:
                # Add padding
                padding = 3
                y_start = 0
                y_end = line_image.shape[0]
                x_start = max(0, current_start - padding)
                x_end = min(line_image.shape[1], char_end + padding)

                char_img = line_image[y_start:y_end, x_start:x_end]

                # Add character
                char_images.append(char_img)
                char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

            in_char = False

    # Handle last character if line ends with a character
    if in_char:
        char_end = line_image.shape[1]
        if char_end - current_start >= min_char_width:
            padding = 3
            y_start = 0
            y_end = line_image.shape[0]
            x_start = max(0, current_start - padding)
            x_end = min(line_image.shape[1], char_end + padding)

            char_img = line_image[y_start:y_end, x_start:x_end]

            char_images.append(char_img)
            char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

    # Check if we need to separate further (more aggressive segmentation)
    if len(char_images) < len(expected_text):
        # Identify wide segments that might contain multiple characters
        wide_segments = []

        for i, (x_start, y_start, w, h) in enumerate(char_positions):
            if w > max_char_width:
                wide_segments.append((i, x_start, x_start + w))

        # Process wide segments
        if wide_segments:
            new_char_images = []
            new_char_positions = []

            for i, (idx, seg_start, seg_end) in enumerate(wide_segments):
                # Try advanced separation
                separation_points = attempt_advanced_separation(line_image, seg_start, seg_end)

                if separation_points:
                    # Add the segments before the first wide segment
                    if i == 0:
                        new_char_images.extend(char_images[:idx])
                        new_char_positions.extend(char_positions[:idx])

                    # Add the previous processed wide segments
                    elif i > 0:
                        prev_idx = wide_segments[i - 1][0]
                        new_char_images.extend(char_images[prev_idx + 1:idx])
                        new_char_positions.extend(char_positions[prev_idx + 1:idx])

                    # Create sub-segments based on separation points
                    all_points = [seg_start] + separation_points + [seg_end]

                    for j in range(len(all_points) - 1):
                        sub_start = all_points[j]
                        sub_end = all_points[j + 1]

                        # Skip if too narrow
                        if sub_end - sub_start < min_char_width:
                            continue

                        # Add padding
                        padding = 3
                        y_start = 0
                        y_end = line_image.shape[0]
                        x_start = max(0, sub_start - padding)
                        x_end = min(line_image.shape[1], sub_end + padding)

                        sub_img = line_image[y_start:y_end, x_start:x_end]

                        new_char_images.append(sub_img)
                        new_char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

                # If no separation points found, keep as is
                else:
                    if i == 0:
                        new_char_images.extend(char_images[:idx + 1])
                        new_char_positions.extend(char_positions[:idx + 1])
                    else:
                        prev_idx = wide_segments[i - 1][0]
                        new_char_images.extend(char_images[prev_idx + 1:idx + 1])
                        new_char_positions.extend(char_positions[prev_idx + 1:idx + 1])

            # Add any remaining segments after the last wide segment
            if wide_segments:
                last_idx = wide_segments[-1][0]
                new_char_images.extend(char_images[last_idx + 1:])
                new_char_positions.extend(char_positions[last_idx + 1:])

            # Update the character lists
            if new_char_images:
                char_images = new_char_images
                char_positions = new_char_positions

    # Check character count against expected
    if len(char_images) != len(expected_text):
        # Try simple division of likely merged characters
        if len(char_images) < len(expected_text):
            difference = len(expected_text) - len(char_images)

            # Find the widest characters that could be merged
            widths = [pos[2] for pos in char_positions]
            avg_width = np.mean(widths)

            # Sort indices by width (descending)
            sorted_indices = np.argsort(widths)[::-1]

            # Attempt to split the widest characters
            new_char_images = []
            new_char_positions = []

            processed_indices = set()
            splits_needed = difference

            for i, char_idx in enumerate(sorted_indices):
                if splits_needed <= 0:
                    break

                x_start, y_start, w, h = char_positions[char_idx]

                # Check if this character is significantly wider than average
                if w > avg_width * 1.5:
                    # Estimate number of characters in this segment
                    est_chars = max(2, round(w / avg_width))

                    if est_chars > 1:
                        # Simple equal division
                        char_width = w // est_chars

                        for j in range(est_chars):
                            sub_x_start = x_start + j * char_width
                            sub_x_end = min(x_start + w, sub_x_start + char_width)

                            sub_img = line_image[y_start:y_start + h, sub_x_start:sub_x_end]

                            # Only add if there's actually ink in this segment
                            if np.sum(sub_img == 255) > 0:
                                new_char_images.append(sub_img)
                                new_char_positions.append((sub_x_start, y_start, sub_x_end - sub_x_start, h))

                        processed_indices.add(char_idx)
                        splits_needed -= (est_chars - 1)

            # Add all unprocessed characters
            for i in range(len(char_images)):
                if i not in processed_indices:
                    new_char_images.append(char_images[i])
                    new_char_positions.append(char_positions[i])

            # Update character lists
            if new_char_images:
                char_images = new_char_images
                char_positions = new_char_positions

    # Optional debug visualization
    if debug:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Line (Expected: "{expected_text}", Found: {len(char_images)} chars)')
        plt.imshow(line_image, cmap='gray')

        plt.subplot(1, 2, 2)
        debug_img = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)

        for i, (x_start, y_start, w, h) in enumerate(char_positions):
            # Use different colors for characters
            color = (0, 255, 0)  # Default green

            # Add expected letter as text
            if i < len(expected_text):
                expected_char = expected_text[i]
                cv2.putText(debug_img, expected_char,
                            (x_start, y_start - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.rectangle(debug_img,
                          (x_start, y_start),
                          (x_start + w, y_start + h),
                          color, 2)

        plt.title(f'Segmented Characters: {len(char_images)}')
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

        # Show individual characters
        if len(char_images) > 0:
            plt.figure(figsize=(15, 3))
            for i, char_img in enumerate(char_images):
                plt.subplot(1, len(char_images), i + 1)
                if i < len(expected_text):
                    plt.title(f'Char {i}: "{expected_text[i]}"')
                else:
                    plt.title(f'Char {i}: Extra')
                plt.imshow(char_img, cmap='gray')
            plt.tight_layout()
            plt.show()

    return char_images, char_positions


# Demonstration function
def demonstrate_segmentation_with_text(image_path, expected_text):
    """
    Demonstrate the segmentation with expected text

    Args:
        image_path (str): Path to the line image
        expected_text (str): Expected text in the image

    Returns:
        tuple: Lists of character images and positions
    """
    # Read the image
    line_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image
    _, binary_image = cv2.threshold(line_image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Segment characters
    char_images, char_positions = segment_with_expected_text(binary_image, expected_text, debug=True)

    dir_path = 'segmented_letters'
    if not os.path.exists(dir_path):
        # Create the directory if it doesn't exist
        os.makedirs(dir_path)

    # Save individual characters
    for i, (char_img, pos) in enumerate(zip(char_images, char_positions)):
        cv2.imwrite(f'{dir_path}/char_{i}{"_" + expected_text[i] if i < len(expected_text) else ""}.png', char_img)
        print(f"Character {i} position: {pos}")

    # Check for missing or extra characters
    if len(char_images) < len(expected_text):
        print(f"WARNING: Missing {len(expected_text) - len(char_images)} characters")
    elif len(char_images) > len(expected_text):
        print(f"WARNING: Found {len(char_images) - len(expected_text)} extra characters")
    else:
        print("Character count matches expected text length!")

    return char_images, char_positions


# Main execution
if __name__ == "__main__":
    # Example usage with expected text
    demonstrate_segmentation_with_text('segmented_lines/line_1.png', "והיה כי יביאך יקוק אל ארץ הכנעני כאשר נשבע לך ולאבתיך ונתנה לך והעברת כל פטר")