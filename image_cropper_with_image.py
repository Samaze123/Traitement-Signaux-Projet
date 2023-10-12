from typing import Final, Tuple, cast
import cv2 as cv
import numpy as np
from classes.Image import Image
from classes.Rectangle import Rectangle

try:
    # ================#
    # GLOBAL VARIABLE #
    # ================#
    CIRCLE_DIAMETER_CM: Final[float] = 0.55
    # IMAGE_PATH: Final[str] = "./tests/tiroir_openlab.png"
    IMAGE_PATH: Final[str] = "./images/rotated_sheet.png"
    TITLE_WINDOW: Final[str] = "Detected Circles"

    opened_image: Image = Image()
    white_rectangle: Rectangle = Rectangle()
    rotated_image: Image = Image()
    rotated_white_rectangle: Rectangle = Rectangle()

    opened_image.max_dimension = 800
    opened_image.original_image = cv.imread(IMAGE_PATH)

    global_selected_circle: Tuple[int] = None

    # ==========#
    # FUNCTIONS #
    # ==========#
    def select_circle(
        event: int, event_x_coord: int, event_y_coord: int, *args
    ) -> None:
        """
        Checks if the click event is inside a circle and performs certain actions if it is.

        Args:
            event (int): The type of event.
            event_x_coord (int): The x-coordinate of the click.
            event_y_coord (int): The y-coordinate of the click.
            *args: Additional arguments.

        Returns:
            None

        Examples:
            ```python
            select_circle(cv.EVENT_LBUTTONDOWN, 10, 20)
            ```
        """

        if event == cv.EVENT_LBUTTONDOWN:
            # Check if the click is inside a circle
            for circle in CIRCLES_IN_WHITE_RECTANGLE:
                center, radius = (circle[0], circle[1]), circle[2]
                if (
                    np.sqrt(
                        (event_x_coord - center[0]) ** 2
                        + (event_y_coord - center[1]) ** 2
                    )
                    < radius
                ):
                    global global_selected_circle
                    global_selected_circle = circle
                    print_informations_circle_rectangle(circle, white_rectangle)
                    change_selected_color(circle)
                    break

    def change_selected_color(selected_circle: np.ndarray) -> None:
        """
        Changes the color of the selected circle in the white rectangle image.

        Args:
            selected_circle (np.ndarray): The coordinates and radius of the selected circle.

        Returns:
            None
        """
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
                        white_rectangle_image.original_image,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (255, 0, 0),
                        2,
                    )
                else:
                    cv.circle(
                        white_rectangle_image.original_image,
                        (X_CIRCLE, Y_CIRCLE),
                        R_CIRCLE,
                        (0, 255, 0),
                        2,
                    )
        cv.imshow(TITLE_WINDOW, white_rectangle_image.original_image)

    def print_informations_circle_rectangle(
        circle: np.ndarray, rectangle: Rectangle
    ) -> None:
        """
        Prints information about a selected circle and rectangle.

        Args:
            circle (np.ndarray): The selected circle represented as an array.
            rectangle (Rectangle): The selected rectangle.

        Returns:
            None

        Examples:
            ```python
            circle = np.array([x, y, radius])
            rectangle = Rectangle(width, height)
            print_informations_circle_rectangle(circle, rectangle)
            ```
        """
        x, y, r = circle
        CIRCLE_DIAMETER_PIXEL: Final[int] = r * 2
        # Calculate and print information about the selected circle immediately
        SCALE_FACTOR: Final[float] = CIRCLE_DIAMETER_CM / CIRCLE_DIAMETER_PIXEL
        WHITE_RECTANGLE_WIDTH_CM: Final[float] = rectangle.size[0] * SCALE_FACTOR
        WHITE_RECTANGLE_HEIGHT_CM: Final[float] = rectangle.size[1] * SCALE_FACTOR

        # Calculate distances
        CIRCLE_DISTANCE_FROM_TOP_PIXEL: Final[int] = y - rectangle.coord[1]
        CIRCLE_DISTANCE_FROM_LEFT_PIXEL: Final[int] = x - rectangle.coord[0]

        CIRCLE_DISTANCE_FROM_TOP_CM = CIRCLE_DISTANCE_FROM_TOP_PIXEL * SCALE_FACTOR
        CIRCLE_DISTANCE_FROM_LEFT_CM = CIRCLE_DISTANCE_FROM_LEFT_PIXEL * SCALE_FACTOR

        print("####################")
        print(
            f"\n=====\nSelected Circle Diameter (in pixels): {CIRCLE_DIAMETER_PIXEL}\n-----\n"
        )
        print(
            f"Selected Rectangle: \
            {WHITE_RECTANGLE_WIDTH_CM:.2f} x {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm"
        )
        print(f"Distance from top (in pixels): {CIRCLE_DISTANCE_FROM_TOP_PIXEL}")
        print(f"Distance from left (in pixels): {CIRCLE_DISTANCE_FROM_LEFT_PIXEL}")
        print(
            f"Distance from top (in centimeters): {CIRCLE_DISTANCE_FROM_TOP_CM:.2f} cm"
        )
        print(
            f"Distance from left (in centimeters): {CIRCLE_DISTANCE_FROM_LEFT_CM:.2f} cm"
        )

    def show(image: np.ndarray) -> None:
        """
        Displays an image using OpenCV and waits for a key press to close the window.

        Args:
            image (np.ndarray): The image to be displayed.

        Returns:
            None
        """

        cv.imshow("", image)
        cv.waitKeyEx(0)
        cv.destroyAllWindows()

    # ========#
    # PROGRAM #
    # ========#

    opened_image.original_dimension = opened_image.original_image.shape[:2]  #  h x w

    # Calculate the new dimensions
    if opened_image.original_dimension[1] > opened_image.original_dimension[0]:
        new_width: int = opened_image.max_dimension
        new_height: int = int(
            opened_image.original_dimension[0]
            * (opened_image.max_dimension / opened_image.original_dimension[1])
        )
    else:
        new_height: int = opened_image.max_dimension
        new_width: int = int(
            opened_image.original_dimension[1]
            * (opened_image.max_dimension / opened_image.original_dimension[0])
        )

    opened_image.resized_image = cv.resize(
        opened_image.original_image, (new_width, new_height)
    )

    opened_image.gray_image = cv.cvtColor(opened_image.resized_image, cv.COLOR_BGR2GRAY)

    opened_image.thresholded_image = cv.threshold(
        opened_image.gray_image, 128, 255, cv.THRESH_BINARY
    )[1]
    white_rectangle.contours = cv.findContours(
        opened_image.thresholded_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0, 0),
    )[0]
    if len(white_rectangle.contours) > 0:
        white_rectangle.largest_contour = max(
            white_rectangle.contours, key=cv.contourArea
        )
        (
            white_rectangle.coord[0],
            white_rectangle.coord[1],
            white_rectangle.size[0],
            white_rectangle.size[1],
        ) = cast(tuple[int, ...], cv.boundingRect(white_rectangle.largest_contour))

        rotated_image.original_image = opened_image.findRotation(white_rectangle)

        rotated_image.gray_image = cv.cvtColor(
            rotated_image.original_image, cv.COLOR_BGR2GRAY
        )

        rotated_image.thresholded_image = cv.threshold(
            rotated_image.gray_image, 128, 255, cv.THRESH_BINARY
        )[1]
        rotated_white_rectangle.contours = cv.findContours(
            rotated_image.thresholded_image,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
            offset=(0, 0),
        )[0]
        if len(rotated_white_rectangle.contours) > 0:
            rotated_white_rectangle.largest_contour = max(
                rotated_white_rectangle.contours, key=cv.contourArea
            )
            cv.drawContours(
                rotated_image.original_image,
                rotated_white_rectangle.contours,
                -1,
                (0, 0, 255),  # BGR : Red
                2,
            )
            (
                rotated_white_rectangle.coord[0],
                rotated_white_rectangle.coord[1],
                rotated_white_rectangle.size[0],
                rotated_white_rectangle.size[1],
            ) = cast(
                tuple[int, ...],
                cv.boundingRect(rotated_white_rectangle.largest_contour),
            )

        white_rectangle_image = Image()

        white_rectangle_image.original_image = rotated_image.original_image[
            rotated_white_rectangle.coord[1] : rotated_white_rectangle.coord[1]
            + rotated_white_rectangle.size[1],
            rotated_white_rectangle.coord[0] : rotated_white_rectangle.coord[0]
            + rotated_white_rectangle.size[0],
        ]

        white_rectangle_image.gray_image = cv.cvtColor(
            white_rectangle_image.original_image, cv.COLOR_BGR2GRAY
        )
        white_rectangle_image.blurred_image = cv.GaussianBlur(
            white_rectangle_image.gray_image, (9, 9), 2
        )

        RAW_CIRCLES_IN_WHITE_RECTANGLE: Final[np.ndarray] = cv.HoughCircles(
            white_rectangle_image.blurred_image,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # Minimum distance between detected circles
            param1=255,  # Upper threshold for edge detection
            param2=13,  # Threshold for circle detection
            minRadius=1,  # Minimum circle radius
            maxRadius=50,  # Maximum circle radius (adjust as needed)
        )

        if RAW_CIRCLES_IN_WHITE_RECTANGLE is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            CIRCLES_IN_WHITE_RECTANGLE: Final[np.ndarray] = np.round(
                RAW_CIRCLES_IN_WHITE_RECTANGLE[0, :]
            ).astype("int")

            # For each circles detected, drawn a solid border around them.
            # Blue for the currently selected and green for the others.
            for i, (
                CIRCLE_X_COORD,
                CIRCLE_Y_CCORD,
                CIRCLE_RADIUS,
            ) in enumerate(CIRCLES_IN_WHITE_RECTANGLE):
                if i == 0:
                    cv.circle(
                        white_rectangle_image.original_image,
                        (CIRCLE_X_COORD, CIRCLE_Y_CCORD),
                        CIRCLE_RADIUS,
                        (255, 0, 0),
                        2,
                    )  # Draw the circle
                else:
                    cv.circle(
                        white_rectangle_image.original_image,
                        (CIRCLE_X_COORD, CIRCLE_Y_CCORD),
                        CIRCLE_RADIUS,
                        (0, 255, 0),
                        2,
                    )  # Draw the circle

            print_informations_circle_rectangle(
                CIRCLES_IN_WHITE_RECTANGLE[0], white_rectangle
            )
            cv.imshow(TITLE_WINDOW, white_rectangle_image.original_image)
            cv.setMouseCallback(TITLE_WINDOW, select_circle)
            while True:
                key_pressed: int = cv.waitKeyEx(0)
                if key_pressed == 27:  # 27 is 'Esc' key
                    break
                if cv.getWindowProperty(TITLE_WINDOW, cv.WND_PROP_VISIBLE) < 1:
                    break

        else:
            print("No circles detected in the image.")

            image1 = cv.resize(
                opened_image.resized_image,
                (opened_image.max_dimension, opened_image.max_dimension),
            )
            image2 = cv.resize(
                white_rectangle_image.original_image,
                (opened_image.max_dimension, opened_image.max_dimension),
            )
            spacer_width = 20  # Adjust the spacer width as needed
            spacer = np.zeros((image1.shape[0], spacer_width, 3), dtype=np.uint8)
            combined_image = np.hstack((image1, spacer, image2))

            # Create a window to display the combined image
            cv.imshow("No Circles", combined_image)
            key_pressed = cv.waitKeyEx(0)

    else:
        print("No white rectangle found in the image.")
        cv.imshow("No Light Rectangle", opened_image.resized_image)
        key_pressed = cv.waitKeyEx(0)

    cv.destroyAllWindows()
    print("EOP")
except Exception as e:
    print(e)
