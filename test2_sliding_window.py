import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation


def find_letter_separation(line_image, start_x, end_x, max_deviation=10):
    """
    Check if there's a potential separation between two letters

    Args:
        line_image (numpy.ndarray): Binary image of the line
        start_x (int): Start x-coordinate of potential letter
        end_x (int): End x-coordinate of potential letter
        max_deviation (int): Maximum allowed vertical deviation

    Returns:
        bool: True if a potential separation is found, False otherwise
    """
    # Crop the region between potential letters
    separation_region = line_image[:, start_x:end_x]

    # Compute vertical projection
    v_projection = np.sum(separation_region == 255, axis=0)

    # Check for potential separation
    # Look for regions with low ink density
    low_ink_threshold = separation_region.shape[0] * 0.1  # 10% of column height
    potential_separations = np.where(v_projection < low_ink_threshold)[0]

    # If we find potential separation points
    if len(potential_separations) > 0:
        return True

    # Advanced separation detection using morphological operations
    # Erode and dilate to remove small connections
    erosion_kernel = np.ones((3, 3), np.uint8)
    separated_region = binary_erosion(separation_region == 255, structure=erosion_kernel)
    separated_region = binary_dilation(separated_region, structure=erosion_kernel)

    # Check if erosion/dilation breaks potential connection
    if np.sum(separated_region) < np.sum(separation_region == 255) * 0.5:
        return True

    return False


def segment_characters_advanced(line_image, debug=False):
    """
    Advanced character segmentation with separation detection

    Args:
        line_image (numpy.ndarray): Binary image of a single text line
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
    max_char_width = line_image.shape[1] // 3  # Maximum character width
    min_white_space = 3  # Minimum white space between characters

    # Lists to store character images and positions
    char_images = []
    char_positions = []

    # Sliding window approach
    current_start = None
    in_char = False
    last_char_end = 0

    for x in range(line_image.shape[1]):
        # Check ink density in vertical slice
        slice_ink_density = np.sum(line_image[:, x] == 255)

        # Detect character start and end
        if slice_ink_density > 0 and not in_char:
            # Check if there's a potential separation from previous character
            if current_start is not None and last_char_end > 0:
                # Check for separation between previous and current character
                if find_letter_separation(line_image, last_char_end, x):
                    # Extract previous character
                    padding = 5
                    y_start = 0
                    y_end = line_image.shape[0]
                    x_start = max(0, current_start - padding)
                    x_end = min(line_image.shape[1], last_char_end + padding)

                    char_img = line_image[y_start:y_end, x_start:x_end]

                    # Additional filtering
                    if np.sum(char_img == 255) > min_char_width * 5:  # Minimum ink area
                        char_images.append(char_img)
                        char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

            # Start of a new character
            current_start = x
            in_char = True

        elif slice_ink_density == 0 and in_char:
            # End of a character
            char_end = x

            # Check if character meets width criteria
            if (char_end - current_start >= min_char_width and
                    char_end - current_start <= max_char_width):

                last_char_end = char_end

                # Extract character image with some padding
                padding = 5
                y_start = 0
                y_end = line_image.shape[0]
                x_start = max(0, current_start - padding)
                x_end = min(line_image.shape[1], char_end + padding)

                char_img = line_image[y_start:y_end, x_start:x_end]

                # Additional filtering
                if np.sum(char_img == 255) > min_char_width * 5:  # Minimum ink area
                    char_images.append(char_img)
                    char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

                # Reset for next character
                in_char = False

    # Handle case of last character if line ends with a character
    if in_char:
        char_end = line_image.shape[1]
        if (char_end - current_start >= min_char_width and
                char_end - current_start <= max_char_width):

            padding = 5
            y_start = 0
            y_end = line_image.shape[0]
            x_start = max(0, current_start - padding)
            x_end = min(line_image.shape[1], char_end + padding)

            char_img = line_image[y_start:y_end, x_start:x_end]

            if np.sum(char_img == 255) > min_char_width * 5:
                char_images.append(char_img)
                char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))

    # Optional debug visualization
    if debug:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Line')
        plt.imshow(line_image, cmap='gray')

        plt.subplot(1, 2, 2)
        debug_img = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)

        for x_start, y_start, w, h in char_positions:
            cv2.rectangle(debug_img,
                          (x_start, y_start),
                          (x_start + w, y_start + h),
                          (0, 255, 0), 2)

        plt.title('Segmented Characters')
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    return char_images, char_positions


# Demonstration function
def demonstrate_segmentation(image_path):
    # Read the image
    line_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image
    _, binary_image = cv2.threshold(line_image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Segment characters
    char_images, char_positions = segment_characters_advanced(binary_image, debug=True)

    dir_path = 'segmented_letters'
    if not os.path.exists(dir_path):
        # Create the directory if it doesn't exist
        os.makedirs(dir_path)
    # Save individual characters
    for i, (char_img, pos) in enumerate(zip(char_images, char_positions)):
        cv2.imwrite(f'{dir_path}/segmented_char_{i}.png', char_img)
        print(f"Character {i} position: {pos}")

    return char_images, char_positions


# Main execution
if __name__ == "__main__":
    demonstrate_segmentation('segmented_lines/line_1.png')