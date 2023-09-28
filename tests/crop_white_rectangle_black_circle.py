import cv2 as cv
import numpy as np

try:
    global_selected_circle = None

    # Define a mouse callback function
    def select_circle(event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            # Check if the click is inside a circle
            for circle in CIRCLES_IN_WHITE_RECTANGLE:
                center, radius = (circle[0], circle[1]), circle[2]
                if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius:
                    global global_selected_circle
                    global_selected_circle = circle
                    print(f"Selected Circle Diameter (in pixels): {circle[2] * 2}")
                    # Calculate and print information about the selected circle immediately
                    SCALE_FACTOR = CIRCLE_DIAMETER_CM / (circle[2] * 2)
                    WHITE_RECTANGLE_WIDTH_CM = WHITE_RECT_WIDTH * SCALE_FACTOR
                    WHITE_RECTANGLE_HEIGHT_CM = WHITE_RECT_HEIGHT * SCALE_FACTOR
                    print(f"Selected Circle Width: {WHITE_RECTANGLE_WIDTH_CM:.2f} cm")
                    print(f"Selected Circle Height: {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm")
                    change_selected_color(circle)
                    break

    def change_selected_color(SELECTED_CIRCLE):
        if SELECTED_CIRCLE is not None:
            for CIRCLE in CIRCLES_IN_WHITE_RECTANGLE:
                X_CIRCLE, Y_CIRCLE, R_CIRCLE = CIRCLE
                (
                    X_SELECTED_CIRCLE,
                    Y_SELECTED_CIRCLE,
                    R_SELECTED_CIRCLE,
                ) = SELECTED_CIRCLE
                if (X_CIRCLE, Y_CIRCLE, R_CIRCLE) == (
                    X_SELECTED_CIRCLE,
                    Y_SELECTED_CIRCLE,
                    R_SELECTED_CIRCLE,
                ):
                    cv.circle(
                        WHITE_RECTANGLE_IMAGE,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (255, 0, 0),
                        2,
                    )  # BGR : Blue
                else:
                    cv.circle(
                        WHITE_RECTANGLE_IMAGE,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (0, 255, 0),
                        2,
                    )  # BGR : Green
        cv.imshow("Detected Circles", WHITE_RECTANGLE_IMAGE)

    CIRCLE_DIAMETER_CM = 0.55
    MAX_DIMENSION = 800
    IMAGE_PATH = "./tests/feuille_blanche.png"

    UNTOUCHED_IMAGE = cv.imread(IMAGE_PATH)

    (
        ORIGINAL_HEIGHT,
        ORIGINAL_WIDTH,
    ) = UNTOUCHED_IMAGE.shape[:2]

    # Calculate the new dimensions
    if ORIGINAL_WIDTH > ORIGINAL_HEIGHT:
        NEW_WIDTH = MAX_DIMENSION
        NEW_HEIGHT = int(ORIGINAL_HEIGHT * (MAX_DIMENSION / ORIGINAL_WIDTH))
    else:
        NEW_HEIGHT = MAX_DIMENSION
        NEW_WIDTH = int(ORIGINAL_WIDTH * (MAX_DIMENSION / ORIGINAL_HEIGHT))

    # Resize the image
    RESIZED_UNTOUCHED_IMAGE = cv.resize(UNTOUCHED_IMAGE, (NEW_WIDTH, NEW_HEIGHT))

    # Convert the image to grayscale
    GRAY_RESIZED_UNTOUCHED_IMAGE = cv.cvtColor(
        RESIZED_UNTOUCHED_IMAGE, cv.COLOR_BGR2GRAY
    )

    # Threshold the image to separate white from black
    THRESHOLDED_UNTOUCHED_IMAGE = cv.threshold(
        GRAY_RESIZED_UNTOUCHED_IMAGE, 128, 255, cv.THRESH_BINARY
    )[1]

    # Find contours in the thresholded image
    WHITE_RECTANGLE_CONTOURS = cv.findContours(
        THRESHOLDED_UNTOUCHED_IMAGE,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0, 0),
    )[0]

    # Assuming there is only one white rectangle, find its bounding box
    if len(WHITE_RECTANGLE_CONTOURS) > 0:
        (
            WHITE_RECT_X_COORD,
            WHITE_RECT_Y_CCORD,
            WHITE_RECT_WIDTH,
            WHITE_RECT_HEIGHT,
        ) = cv.boundingRect(WHITE_RECTANGLE_CONTOURS[5])

        # Crop the image to the bounding box
        WHITE_RECTANGLE_IMAGE = RESIZED_UNTOUCHED_IMAGE[
            WHITE_RECT_Y_CCORD : WHITE_RECT_Y_CCORD + WHITE_RECT_HEIGHT,
            WHITE_RECT_X_COORD : WHITE_RECT_X_COORD + WHITE_RECT_WIDTH,
        ]
        cv.rectangle(
            WHITE_RECTANGLE_IMAGE,
            (0, 0),
            (WHITE_RECT_WIDTH - 2, WHITE_RECT_HEIGHT - 2),
            (0, 0, 255),
            2,
        )
        # Convert the cropped image to grayscale
        GRAY_WHITE_RECTANGLE = cv.cvtColor(WHITE_RECTANGLE_IMAGE, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve circle detection
        BLURRED_GRAY_WHITE_RECTANGLE = cv.GaussianBlur(GRAY_WHITE_RECTANGLE, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles in the image
        RAW_CIRCLES_IN_WHITE_RECTANGLE = cv.HoughCircles(
            BLURRED_GRAY_WHITE_RECTANGLE,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=20,  # Minimum distance between detected circles
            param1=50,  # Upper threshold for edge detection
            param2=15,  # Threshold for circle detection
            minRadius=5,  # Minimum circle radius
            maxRadius=50,  # Maximum circle radius (adjust as needed)
        )

        # Check if any circles were found
        if RAW_CIRCLES_IN_WHITE_RECTANGLE is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            CIRCLES_IN_WHITE_RECTANGLE = np.round(
                RAW_CIRCLES_IN_WHITE_RECTANGLE[0, :]
            ).astype("int")
            # Loop over the circles and draw them on the image
            for (
                CIRCLE_X_COORD,
                CIRCLE_Y_CCORD,
                CIRCLE_RADIUS,
            ) in CIRCLES_IN_WHITE_RECTANGLE:
                cv.circle(
                    WHITE_RECTANGLE_IMAGE,
                    (CIRCLE_X_COORD, CIRCLE_Y_CCORD),
                    CIRCLE_RADIUS,
                    (0, 255, 0),
                    2,
                )  # Draw the circle

            # Measure the diameter of the first detected circle
            CIRCLE_DIAMETER_PIXELS = CIRCLES_IN_WHITE_RECTANGLE[0][2] * 2

            print(f"Black Circle Diameter (in pixels): {CIRCLE_DIAMETER_PIXELS}")
            # Calculate the scale factor from pixels to centimeters
            SCALE_FACTOR = CIRCLE_DIAMETER_CM / CIRCLE_DIAMETER_PIXELS

            print(f"White Rectangle Width: {(WHITE_RECT_WIDTH * SCALE_FACTOR):.2f} cm")
            print(
                f"White Rectangle Height: {(WHITE_RECT_HEIGHT * SCALE_FACTOR):.2f} cm"
            )

            # Display the image with detected circles
            # cv.imshow("Detected Circles", WHITE_RECTANGLE)
            cv.imshow("Detected Circles", WHITE_RECTANGLE_IMAGE)
            cv.setMouseCallback("Detected Circles", select_circle)

            while True:
                KEY = cv.waitKeyEx(0)
                if KEY == 27:  # Press 'Esc' to exit the program
                    break
                if cv.getWindowProperty("Detected Circles", cv.WND_PROP_VISIBLE) < 1:
                    break
        else:
            print("No circles detected in the image.")
    else:
        print("No white rectangle found in the image.")

    cv.destroyAllWindows()
    print("EOP")
except Exception as e:
    print(e)
