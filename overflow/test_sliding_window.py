import numpy as np
import cv2
import matplotlib.pyplot as plt


def segment_characters_sliding_window(line_image, debug=False):
    """
    Segment characters using a sliding window approach

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

    for x in range(line_image.shape[1]):
        # Check ink density in vertical slice
        slice_ink_density = np.sum(line_image[:, x] == 255)

        # Detect character start and end
        if slice_ink_density > 0 and not in_char:
            # Start of a character
            current_start = x
            in_char = True

        elif slice_ink_density == 0 and in_char:
            # End of a character
            char_end = x

            # Check if character meets width criteria
            if (char_end - current_start >= min_char_width and
                    char_end - current_start <= max_char_width):

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


def adaptive_segmentation(line_image, debug=False):
    """
    Adaptive character segmentation with multiple strategies

    Args:
        line_image (numpy.ndarray): Binary image of a single text line
        debug (bool): Whether to show debug visualization

    Returns:
        tuple: List of character images and their positions
    """
    # Try sliding window segmentation
    char_images, char_positions = segment_characters_sliding_window(line_image, debug)

    # If no characters found, try alternative methods
    if len(char_images) == 0:
        # Fallback to connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_image, connectivity=8
        )

        # Filter and extract components
        char_images = []
        char_positions = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # Basic filtering
            if (10 < w < line_image.shape[1] // 3 and
                    h < line_image.shape[0] and
                    area > 50):
                char_mask = np.zeros_like(line_image)
                char_mask[labels == i] = 255

                char_images.append(char_mask)
                char_positions.append((x, y, w, h))

    return char_images, char_positions


# Example usage
def main():
    # Load a line image (replace with your image loading method)
    line_image = cv2.imread('../segmented_lines/line_1.png', cv2.IMREAD_GRAYSCALE)

    # Binarize if needed
    _, line_image = cv2.threshold(line_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Segment characters
    char_images, char_positions = adaptive_segmentation(line_image, debug=True)

    # Optionally process or save characters
    for i, (char_img, pos) in enumerate(zip(char_images, char_positions)):
        cv2.imwrite(f'segmented_letters/char_{i}.png', char_img)
        print(f"Character {i} position: {pos}")


if __name__ == "__main__":
    main()