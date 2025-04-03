import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.measure import regionprops
from skimage.filters import sobel


def find_letter_separation(line_image, start_x, end_x, max_deviation=10):
    """
    Enhanced function to check for separations between letters, including slanted/curved connections
    
    Args:
        line_image (numpy.ndarray): Binary image of the line
        start_x (int): Start x-coordinate of potential letter
        end_x (int): End x-coordinate of potential letter
        max_deviation (int): Maximum allowed vertical deviation
    
    Returns:
        bool: True if a potential separation is found, False otherwise
    """
    # Ensure we have a valid region to analyze
    if end_x <= start_x or start_x < 0 or end_x >= line_image.shape[1]:
        return False
        
    # Crop the region between potential letters
    separation_region = line_image[:, start_x:end_x]
    region_height = separation_region.shape[0]
    region_width = separation_region.shape[1]
    
    # If the region is too narrow, it's likely not a separation
    if region_width < 2:
        return False
    
    # 1. First check: Calculate ink density
    v_projection = np.sum(separation_region == 255, axis=0)
    low_ink_threshold = region_height * 0.1  # 10% of column height
    potential_separations = np.where(v_projection < low_ink_threshold)[0]
    
    if len(potential_separations) > 0:
        return True
    
    # 2. Second check: Detect thin connections using morphological operations
    erosion_kernel = np.ones((2, 1), np.uint8)  # Vertical kernel to target horizontal connections
    separated_region = binary_erosion(separation_region == 255, structure=erosion_kernel)
    
    # If erosion significantly reduces the pixel count, it suggests thin connections
    if np.sum(separated_region) < np.sum(separation_region == 255) * 0.7:
        return True
    
    # 3. Third check: Edge detection to find character boundaries
    # Apply Sobel edge detection to find vertical edges
    edges = sobel(separation_region == 255)
    edge_threshold = np.max(edges) * 0.5
    strong_edges = edges > edge_threshold
    
    # If we detect strong vertical edges, likely indicates character boundaries
    if np.sum(strong_edges) > region_height * 0.3:
        return True
    
    # 4. Fourth check: Check for slanted or curved separations
    # Analyze the ink density distribution
    # For slanted connections, the ink density should change along the x-axis
    # Calculate the gradient of ink density along columns
    if region_width > 3:
        gradient = np.abs(np.diff(v_projection))
        if np.max(gradient) > region_height * 0.2:
            return True
    
    # 5. Fifth check: Connected component analysis after targeted erosion
    # Use a more aggressive erosion specifically aimed at breaking narrow connections
    targeted_kernel = np.ones((1, 2), np.uint8)  # Horizontal kernel to target vertical connections
    aggressively_eroded = binary_erosion(separation_region == 255, structure=targeted_kernel, iterations=2)
    
    # Count connected components
    labeled_array, num_features = label(aggressively_eroded)
    
    # If erosion creates multiple components, it suggests a separation point
    if num_features > 1:
        return True
    
    return False

def segment_characters_advanced(line_image, debug=False):
    """
    Advanced character segmentation with improved separation detection
    
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
    
    # Apply a slight dilation to enhance character boundaries
    kernel = np.ones((2, 2), np.uint8)
    processed_image = cv2.dilate(line_image, kernel, iterations=1)
    
    # Compute vertical projection (ink density)
    v_projection = np.sum(processed_image == 255, axis=0)
    
    # Detect potential segmentation points using projection profile
    # Calculate gradient of projection to find significant changes
    gradient = np.abs(np.diff(v_projection))
    
    # Parameters
    min_char_width = 8  # Minimum character width
    max_char_width = line_image.shape[1] // 3  # Maximum character width
    min_white_space = 2  # Minimum white space between characters
    gradient_threshold = np.mean(gradient) + np.std(gradient) * 0.7  # Adaptive threshold for gradient

    # Find potential segmentation points
    potential_cuts = np.where(gradient > gradient_threshold)[0]
    
    # Enhance with local minima in the projection
    local_minima = []
    for i in range(1, len(v_projection) - 1):
        if v_projection[i] < v_projection[i-1] and v_projection[i] < v_projection[i+1]:
            # Check if it's a significant minimum
            if v_projection[i] < np.mean(v_projection) * 0.7:
                local_minima.append(i)
    
    # Combine gradients and local minima for better segmentation points
    all_potential_cuts = np.unique(np.concatenate((potential_cuts, np.array(local_minima))))
    
    # Lists to store character images and positions
    char_images = []
    char_positions = []
    
    # Initial segmentation based on potential cuts
    if len(all_potential_cuts) > 0:
        segments = []
        
        # Add start and end points
        cut_points = [0] + list(all_potential_cuts) + [line_image.shape[1]-1]
        cut_points = sorted(list(set(cut_points)))  # Ensure unique and sorted
        
        # Create initial segments
        for i in range(len(cut_points) - 1):
            segment_start = cut_points[i]
            segment_end = cut_points[i + 1]
            
            # Check if segment is wide enough
            if segment_end - segment_start >= min_char_width:
                segments.append((segment_start, segment_end))
        
        # Further refinement of segments
        refined_segments = []
        for start, end in segments:
            width = end - start
            
            # Skip very narrow segments
            if width < min_char_width:
                continue
                
            # Further split wide segments
            if width > max_char_width:
                # Check for potential separations within the segment
                mid_point = start + width // 2
                
                # Check for separation in middle third
                third_width = width // 3
                check_start = start + third_width
                check_end = end - third_width
                
                found_split = False
                for x in range(check_start, check_end):
                    # Check if this is a good separation point
                    if find_letter_separation(processed_image, x-2, x+2):
                        refined_segments.append((start, x))
                        refined_segments.append((x, end))
                        found_split = True
                        break
                
                if not found_split:
                    refined_segments.append((start, end))
            else:
                refined_segments.append((start, end))
        
        # Extract character images based on refined segments
        for i, (start, end) in enumerate(refined_segments):
            # Skip segments that are too narrow
            if end - start < min_char_width:
                continue
            
            # Add padding
            padding = 3
            y_start = 0
            y_end = line_image.shape[0]
            x_start = max(0, start - padding)
            x_end = min(line_image.shape[1], end + padding)
            
            char_img = line_image[y_start:y_end, x_start:x_end]
            
            # Filter out segments with too little ink (likely spaces)
            min_ink_threshold = min_char_width * min_char_width * 0.5
            if np.sum(char_img == 255) > min_ink_threshold:
                char_images.append(char_img)
                char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))
    
    # If no characters were found with the refined approach, fall back to the original method
    if len(char_images) == 0:
        # Sliding window approach (similar to original code)
        current_start = None
        in_char = False
        last_char_end = 0
        
        for x in range(line_image.shape[1]):
            # Check ink density in vertical slice
            slice_ink_density = np.sum(line_image[:, x] == 255)
            
            # Detect character start and end
            if slice_ink_density > 0 and not in_char:
                if current_start is not None and last_char_end > 0:
                    if find_letter_separation(line_image, last_char_end, x):
                        padding = 3
                        y_start = 0
                        y_end = line_image.shape[0]
                        x_start = max(0, current_start - padding)
                        x_end = min(line_image.shape[1], last_char_end + padding)
                        
                        char_img = line_image[y_start:y_end, x_start:x_end]
                        if np.sum(char_img == 255) > min_char_width * 5:
                            char_images.append(char_img)
                            char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))
                
                current_start = x
                in_char = True
            
            elif slice_ink_density == 0 and in_char:
                char_end = x
                
                if char_end - current_start >= min_char_width and char_end - current_start <= max_char_width:
                    last_char_end = char_end
                    
                    padding = 3
                    y_start = 0
                    y_end = line_image.shape[0]
                    x_start = max(0, current_start - padding)
                    x_end = min(line_image.shape[1], char_end + padding)
                    
                    char_img = line_image[y_start:y_end, x_start:x_end]
                    if np.sum(char_img == 255) > min_char_width * 5:
                        char_images.append(char_img)
                        char_positions.append((x_start, y_start, x_end - x_start, y_end - y_start))
                
                in_char = False
        
        # Handle last character
        if in_char:
            char_end = line_image.shape[1]
            if char_end - current_start >= min_char_width and char_end - current_start <= max_char_width:
                padding = 3
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
        plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.title('Original Line')
        plt.imshow(line_image, cmap='gray')
        
        # Segmented characters
        plt.subplot(2, 2, 2)
        debug_img = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)
        
        for x_start, y_start, w, h in char_positions:
            cv2.rectangle(debug_img,
                          (x_start, y_start),
                          (x_start + w, y_start + h),
                          (0, 255, 0), 2)
        
        plt.title('Segmented Characters')
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        
        # Vertical projection
        plt.subplot(2, 2, 3)
        plt.title('Vertical Projection')
        plt.plot(v_projection)
        plt.axhline(y=np.mean(v_projection) * 0.7, color='r', linestyle='-', label='Threshold')
        
        # Gradient of projection
        plt.subplot(2, 2, 4)
        plt.title('Gradient of Projection')
        plt.plot(np.arange(len(gradient)), gradient)
        plt.axhline(y=gradient_threshold, color='r', linestyle='-', label='Threshold')
        
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
