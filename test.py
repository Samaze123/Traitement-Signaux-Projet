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

    # Label connected components in the thresholded image
    labels = measure.label(thresholded, connectivity=2)

    # Check if there are any labeled regions (white regions)
    if np.max(labels) > 0:
        # Find the largest connected white region
        largest_region_label = np.argmax(np.bincount(labels.flat)[1:]) + 1

        # Create a mask to isolate the largest white region
        mask = labels == largest_region_label

        # Apply the mask to the original image
        result = np.copy(image)
        result[~mask] = 0

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Initial Image")
        axes[0].axis("off")
        axes[1].imshow(result)
        axes[1].set_title("Result")
        axes[1].axis("off")
    else:
        # Handle the case when no white regions are found
        print("No white regions found in the image.")
        plt.imshow(image)
        plt.axis("off")
    plt.show()
except Exception as e:
    print(e)
