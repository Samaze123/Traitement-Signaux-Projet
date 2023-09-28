import cv2 as cv
import numpy as np
from classes.Circle import Circle
from classes.Image import Image

try:
    # ================#
    # GLOBAL VARIABLE #
    # ================#
    CIRCLE_DIAMETER_CM = 4
    MAX_DIMENSION = 800
    IMAGE_PATH = "./tests/tiroir_openlab.png"
    TITLE_WINDOW = "Detected Circles"

    circle = Circle()
    circle.diameter(CIRCLE_DIAMETER_CM)
    image = Image(IMAGE_PATH, MAX_DIMENSION, TITLE_WINDOW)

    image.read()

    global_selected_circle = None

    # ==========#
    # FUNCTIONS #
    # ==========#
    def select_circle(event, x, y, *args):
        if event == cv.EVENT_LBUTTONDOWN:
            # Check if the click is inside a circle
            for CIRCLE in CIRCLES_IN_WHITE_RECTANGLE:
                CENTER, RADIUS = (CIRCLE[0], CIRCLE[1]), CIRCLE[2]
                if np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) < RADIUS:
                    global global_selected_circle
                    global_selected_circle = CIRCLE
                    print_informations_circle_rectangle(
                        CIRCLE, WHITE_RECT_WIDTH, WHITE_RECT_HEIGHT
                    )
                    change_selected_color(CIRCLE)
                    break

    def change_selected_color(selected_circle):
        if selected_circle is not None:
            for CIRCLE in CIRCLES_IN_WHITE_RECTANGLE:
                X_CIRCLE, Y_CIRCLE, R_CIRCLE = CIRCLE
                (
                    X_SELECTED_CIRCLE,
                    Y_SELECTED_CIRCLE,
                    R_SELECTED_CIRCLE,
                ) = selected_circle
                if (X_CIRCLE, Y_CIRCLE, R_CIRCLE) == (
                    X_SELECTED_CIRCLE,
                    Y_SELECTED_CIRCLE,
                    R_SELECTED_CIRCLE,
                ):
                    cv.circle(
                        WHITE_RECT_IMAGE,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (255, 0, 0),
                        2,
                    )
                else:
                    cv.circle(
                        WHITE_RECT_IMAGE,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (0, 255, 0),
                        2,
                    )
        cv.imshow(TITLE_WINDOW, WHITE_RECT_IMAGE)

    def print_informations_circle_rectangle(circle, rect_width, rect_height):
        print("####################")
        print(
            f"\n=====\nSelected Circle Diameter (in pixels): {circle[2] * 2}\n-----\n"
        )
        # Calculate and print information about the selected circle immediately
        SCALE_FACTOR = CIRCLE_DIAMETER_CM / (circle[2] * 2)
        WHITE_RECTANGLE_WIDTH_CM = rect_width * SCALE_FACTOR
        WHITE_RECTANGLE_HEIGHT_CM = rect_height * SCALE_FACTOR
        print(
            f"Selected Rectangle: {WHITE_RECTANGLE_WIDTH_CM:.2f} x {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm"
        )
        # print(f"Selected Rectangle Height: {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm")

    # ========#
    # PROGRAM #
    # ========#
    image.save_dimension(2)

    image.resize()

    image.grayscale()

    image.threshold()

    WHITE_RECTANGLE_CONTOURS = cv.findContours(
        image.thresholded_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0, 0),
    )[0]

    if len(WHITE_RECTANGLE_CONTOURS) > 0:
        largest_area = 0
        largest_rect = (0, 0, 0, 0)
        for CONTOUR in WHITE_RECTANGLE_CONTOURS:
            X_CONTOUR, Y_CONTOUR, WIDTH_CONTOUR, HEIGHT_CONTOUR = cv.boundingRect(
                CONTOUR
            )
            AREA = WIDTH_CONTOUR * HEIGHT_CONTOUR
            if AREA > largest_area:
                largest_area = AREA
                largest_rect = (X_CONTOUR, Y_CONTOUR, WIDTH_CONTOUR, HEIGHT_CONTOUR)

        (
            WHITE_RECT_X_COORD,
            WHITE_RECT_Y_COORD,
            WHITE_RECT_WIDTH,
            WHITE_RECT_HEIGHT,
        ) = largest_rect

        WHITE_RECT_IMAGE = image.untouched_image[
            WHITE_RECT_Y_COORD : WHITE_RECT_Y_COORD + WHITE_RECT_HEIGHT,
            WHITE_RECT_X_COORD : WHITE_RECT_X_COORD + WHITE_RECT_WIDTH,
        ]
        cv.rectangle(
            WHITE_RECT_IMAGE,
            (0, 0),
            (WHITE_RECT_WIDTH - 2, WHITE_RECT_HEIGHT - 2),
            (0, 0, 255),
            2,
        )
        GRAY_WHITE_RECTANGLE = cv.cvtColor(WHITE_RECT_IMAGE, cv.COLOR_BGR2GRAY)

        BLURRED_GRAY_WHITE_RECTANGLE = cv.GaussianBlur(GRAY_WHITE_RECTANGLE, (9, 9), 2)

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

        if RAW_CIRCLES_IN_WHITE_RECTANGLE is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            CIRCLES_IN_WHITE_RECTANGLE = np.round(
                RAW_CIRCLES_IN_WHITE_RECTANGLE[0, :]
            ).astype("int")

            # For each circles detected, drawn a solid border around them. Blue for the currently selected and green for the others.
            for i, (
                CIRCLE_X_COORD,
                CIRCLE_Y_CCORD,
                CIRCLE_RADIUS,
            ) in enumerate(CIRCLES_IN_WHITE_RECTANGLE):
                if i == 0:
                    cv.circle(
                        WHITE_RECT_IMAGE,
                        (CIRCLE_X_COORD, CIRCLE_Y_CCORD),
                        CIRCLE_RADIUS,
                        (255, 0, 0),
                        2,
                    )  # Draw the circle
                else:
                    cv.circle(
                        WHITE_RECT_IMAGE,
                        (CIRCLE_X_COORD, CIRCLE_Y_CCORD),
                        CIRCLE_RADIUS,
                        (0, 255, 0),
                        2,
                    )  # Draw the circle

            print_informations_circle_rectangle(
                CIRCLES_IN_WHITE_RECTANGLE[0], WHITE_RECT_WIDTH, WHITE_RECT_HEIGHT
            )

            cv.imshow(TITLE_WINDOW, WHITE_RECT_IMAGE)
            cv.setMouseCallback(TITLE_WINDOW, select_circle)
            while True:
                KEY = cv.waitKeyEx(0)
                if KEY == 27:  # 27 is 'Esc' key
                    break
                if cv.getWindowProperty(TITLE_WINDOW, cv.WND_PROP_VISIBLE) < 1:
                    break

        else:
            print("No circles detected in the image.")
            cv.imshow("No Circles", image.gray_resized_image)
            KEY = cv.waitKeyEx(0)

    else:
        print("No white rectangle found in the image.")
        cv.imshow("No Light Rectangle", image.gray_resized_image)
        KEY = cv.waitKeyEx(0)

    cv.destroyAllWindows()
    print("EOP")
except Exception as e:
    print(e)
