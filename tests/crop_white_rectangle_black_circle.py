import cv2 as cv
import numpy as np

try:
    CIRCLE_DIAMETER_CM = 10
    MAX_DIMENSION = 800

    image_path = "./tests/feuille_blanche.png"
    untouched_image = cv.imread(image_path)

    original_height, original_width = untouched_image.shape[:2]

    # Calculate the new dimensions
    if original_width > original_height:
        new_width = MAX_DIMENSION
        new_height = int(original_height * (MAX_DIMENSION / original_width))
    else:
        new_height = MAX_DIMENSION
        new_width = int(original_width * (MAX_DIMENSION / original_height))

    # Resize the image
    resized_untouched_image = cv.resize(untouched_image, (new_width, new_height))

    # Convert the image to grayscale
    gray_resized_untouched_image = cv.cvtColor(
        resized_untouched_image, cv.COLOR_BGR2GRAY
    )

    # Threshold the image to separate white from black
    thresholded_untouched_image = cv.threshold(
        gray_resized_untouched_image, 128, 255, cv.THRESH_BINARY
    )[1]

    # cv.imshow("Detected Circles", thresholded_untouched_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # exit()

    # Find contours in the thresholded image
    white_rectangle_contours = cv.findContours(
        thresholded_untouched_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0, 0),
    )[0]

    # Assuming there is only one white rectangle, find its bounding box
    if len(white_rectangle_contours) > 0:
        x, y, w, h = cv.boundingRect(white_rectangle_contours[0])
        # Draw the bounding rectangle on the image
        image_with_bbox = cv.cvtColor(
            resized_untouched_image, cv.COLOR_GRAY2BGR
        )  # Convert to a color image
        cv.rectangle(
            image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2
        )  # Draw the bounding rectangle

        # Display the image with the bounding rectangle
        cv.imshow("Bounding Rectangle", image_with_bbox)
        cv.waitKey(0)
        cv.destroyAllWindows()
        exit()
        # Crop the image to the bounding box
        white_rectangle = resized_untouched_image[y : y + h, x : x + w]
        cv.rectangle(white_rectangle, (0, 0), (w - 2, h - 2), (0, 0, 255), 2)
        # Convert the cropped image to grayscale
        gray_white_rectangle = cv.cvtColor(white_rectangle, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve circle detection
        gray_white_rectangle = cv.GaussianBlur(gray_white_rectangle, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles in the image
        circles_in_white_rectangle = cv.HoughCircles(
            gray_white_rectangle,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=1,  # Minimum distance between detected circles
            param1=50,  # Upper threshold for edge detection
            param2=15,  # Threshold for circle detection
            minRadius=5,  # Minimum circle radius
            maxRadius=25,  # Maximum circle radius (adjust as needed)
        )

        # Check if any circles were found
        if circles_in_white_rectangle is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles_in_white_rectangle = np.round(
                circles_in_white_rectangle[0, :]
            ).astype("int")
            # Loop over the circles and draw them on the image
            for x, y, r in circles_in_white_rectangle:
                cv.circle(white_rectangle, (x, y), r, (0, 255, 0), 2)  # Draw the circle

            # Measure the diameter of the first detected circle
            circle_diameter_pixels = circles_in_white_rectangle[0][2] * 2

            print(f"Black Circle Diameter (in pixels): {circle_diameter_pixels}")
            # Calculate the scale factor from pixels to centimeters
            scale_factor = CIRCLE_DIAMETER_CM / circle_diameter_pixels

            # Measure the dimensions of the white rectangle in pixels
            white_rectangle_width_pixels = w  # Measure the width in pixels
            white_rectangle_height_pixels = h  # Measure the height in pixels

            # Convert the pixel dimensions to centimeters using the scale factor
            white_rectangle_width_cm = white_rectangle_width_pixels * scale_factor
            white_rectangle_height_cm = white_rectangle_height_pixels * scale_factor

            print(f"White Rectangle Width: {white_rectangle_width_cm:.2f} cm")
            print(f"White Rectangle Height: {white_rectangle_height_cm:.2f} cm")

            # Display the image with detected circles
            cv.imshow("Detected Circles", white_rectangle)
            cv.waitKey(0)

        else:
            print("No circles detected in the image.")
    else:
        print("No white rectangle found in the image.")

    # Release any OpenCV resources (not always necessary, but good practice)
    cv.destroyAllWindows()
except Exception as e:
    print(e)
