import cv2
import numpy as np

def gaussian_blur_filter(filter_size, sigma):
    """
    Generates a Gaussian blur filter kernel.
    
    Args:
        filter_size (int): Size of the filter (e.g., 9 for a 9x9 filter).
        sigma (float): Sigma value for the Gaussian function.
    
    Returns:
        np.ndarray: Gaussian blur filter kernel.
    """
    # Create an empty kernel
    kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
    
    # Calculate the center of the kernel
    center = filter_size // 2
    
    # Compute Gaussian weights
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel so that the sum of weights equals 1
    kernel /= np.sum(kernel)
    
    return kernel

def apply_gaussian_blur(image, filter_size, sigma):
    """
    Applies a Gaussian blur to the input image.
    
    Args:
        image (np.ndarray): Input image.
        filter_size (int): Size of the filter (e.g., 9 for a 9x9 filter).
        sigma (float): Sigma value for the Gaussian function.
    
    Returns:
        np.ndarray: Blurred image.
    """
    # Generate the Gaussian blur filter
    kernel = gaussian_blur_filter(filter_size, sigma)
    
    # Apply the filter to the image using OpenCV's filter2D function
    blurred_image = cv2.filter2D(image, -1, kernel)
    
    return blurred_image

def main():
    # Load the input image
    input_image_path = "mine3.png"
    output_image_path = "pythono.jpg"
    
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return
    
    # Define the filter size and sigma
    filter_size = 15
    sigma = 15
    
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image, filter_size, sigma)
    
    # Save the output image
    cv2.imwrite(output_image_path, blurred_image)
    print(f"Blurred image saved to {output_image_path}")

if __name__ == "__main__":
    main()