import cv2 as cv
import numpy as np
import os

try:
    MAX_DIMENSION = 800
    # IMAGE_PATH = "./tests/feuille_blanche.png"
    IMAGE_PATH = "./images/images_converted"

    # Create a list to store all the resized images
    resized_images = []

    for i, filename in enumerate(os.listdir(IMAGE_PATH)):
        print(filename)
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(IMAGE_PATH, filename)
            UNTOUCHED_IMAGE = cv.imread(image_path)

            if UNTOUCHED_IMAGE is None:
                print(f"Error: Could not load image {image_path}")
                continue

            (
                ORIGINAL_HEIGHT,
                ORIGINAL_WIDTH,
            ) = UNTOUCHED_IMAGE.shape[:2]

            # Check if the image needs to be resized
            if ORIGINAL_WIDTH > MAX_DIMENSION or ORIGINAL_HEIGHT > MAX_DIMENSION:
                # Calculate the new dimensions
                if ORIGINAL_WIDTH > ORIGINAL_HEIGHT:
                    NEW_WIDTH = MAX_DIMENSION
                    NEW_HEIGHT = int(ORIGINAL_HEIGHT * (MAX_DIMENSION / ORIGINAL_WIDTH))
                else:
                    NEW_HEIGHT = MAX_DIMENSION
                    NEW_WIDTH = int(ORIGINAL_WIDTH * (MAX_DIMENSION / ORIGINAL_HEIGHT))

                # Resize the image
                RESIZED_UNTOUCHED_IMAGE = cv.resize(
                    UNTOUCHED_IMAGE, (NEW_WIDTH, NEW_HEIGHT)
                )
            else:
                RESIZED_UNTOUCHED_IMAGE = UNTOUCHED_IMAGE

            # Append the resized image to the list
            resized_images.append(RESIZED_UNTOUCHED_IMAGE)
    if len(resized_images) != 0:
        # Create a numpy array to store all the resized images
        combined_image = np.zeros(
            (
                MAX_DIMENSION * ((len(resized_images) - 1) // 3 + 1),
                MAX_DIMENSION * min(3, len(resized_images)),
                3,
            ),
            dtype=np.uint8,
        )

        # Combine all the resized images into a single numpy array
        for i, image in enumerate(resized_images):
            x = i % 3 * MAX_DIMENSION
            y = i // 3 * MAX_DIMENSION
            combined_image[y : y + image.shape[0], x : x + image.shape[1]] = image

        # Display the combined image
        if combined_image.size > 0:
            cv.imshow("Resized Images", combined_image)
            cv.waitKeyEx(0)
            cv.destroyAllWindows()
        else:
            print("Error: Could not load image.")
    print("resized_images", len(resized_images))
except Exception as e:
    print(e)
    cv.waitKeyEx(0)
    cv.destroyAllWindows()

print("EOP")
