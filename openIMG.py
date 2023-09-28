import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

try:
    # Load the image
    image = plt.imread("./example1.png")

    # Convert the image to grayscale
    gray = np.mean(image, axis=2)

    # Threshold the image to extract the white regions
    thresholded = gray > 200

    # Find contours in the thresholded image using skimage
    contours = measure.find_contours(thresholded, 0.5)

    # Find the largest white rectangle
    largest_area = 0
    largest_rectangle = None

    for contour in contours:
        if contour.size > 0:  # Check if the contour is not empty
            area = np.prod(contour.shape, axis=1).max()  # Calculate area using shape
            if area > largest_area:
                largest_area = area
                largest_rectangle = contour

    # Create a mask for the largest white rectangle using NumPy
    mask = np.zeros_like(gray, dtype=bool)
    if largest_rectangle is not None:
        for vertices in largest_rectangle:
            x, y = vertices.astype(int)
            mask[y, x] = True

    # Apply the mask to the original image
    result = np.copy(image)
    result[~mask] = 0

    # Display the initial image and the result side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Initial Image")
    axes[0].axis("off")
    axes[1].imshow(result)
    axes[1].set_title("Result")
    axes[1].axis("off")
    plt.show()

except Exception as e:
    print(e)
