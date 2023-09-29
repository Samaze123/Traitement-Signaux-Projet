from typing import Final, Tuple, cast
import cv2 as cv
import numpy as np

try:
    # ================#
    # GLOBAL VARIABLE #
    # ================#
    CIRCLE_DIAMETER_CM: Final[float] = 0.55
    MAX_DIMENSION: Final[int] = 800
    # IMAGE_PATH: Final[str] = "./tests/tiroir_openlab.png"
    IMAGE_PATH: Final[str] = "./tests/rotated_sheet.png"
    TITLE_WINDOW: Final[str] = "Detected Circles"

    UNTOUCHED_IMAGE: Final[np.ndarray] = cv.imread(IMAGE_PATH)

    global_selected_circle: [None | Tuple[int]] = None

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
            x (int): The x-coordinate of the click.
            y (int): The y-coordinate of the click.
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
                    print_informations_circle_rectangle(
                        circle, WHITE_RECT_WIDTH, WHITE_RECT_HEIGHT
                    )
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

    def print_informations_circle_rectangle(
        circle: np.ndarray, rect_width: int, rect_height: int
    ) -> None:
        """
        Prints information about the selected circle and the corresponding rectangle.

        Args:
            circle (np.ndarray): The coordinates and radius of the selected circle.
            rect_width (int): The width of the rectangle.
            rect_height (int): The height of the rectangle.

        Returns:
            None
        """

        print("####################")
        print(
            f"\n=====\nSelected Circle Diameter (in pixels): {circle[2] * 2}\n-----\n"
        )
        # Calculate and print information about the selected circle immediately
        SCALE_FACTOR: Final[float] = CIRCLE_DIAMETER_CM / (circle[2] * 2)
        WHITE_RECTANGLE_WIDTH_CM: Final[float] = rect_width * SCALE_FACTOR
        WHITE_RECTANGLE_HEIGHT_CM: Final[float] = rect_height * SCALE_FACTOR
        print(
            f"Selected Rectangle: \
            {WHITE_RECTANGLE_WIDTH_CM:.2f} x {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm"
        )
        # print(f"Selected Rectangle Height: {WHITE_RECTANGLE_HEIGHT_CM:.2f} cm")

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

    ORIGINAL_HEIGHT: Final[int] = UNTOUCHED_IMAGE.shape[0]
    ORIGINAL_WIDTH: Final[int] = UNTOUCHED_IMAGE.shape[1]

    # Calculate the new dimensions
    if ORIGINAL_WIDTH > ORIGINAL_HEIGHT:
        new_width: int = MAX_DIMENSION
        new_height: int = int(ORIGINAL_HEIGHT * (MAX_DIMENSION / ORIGINAL_WIDTH))
    else:
        new_height: int = MAX_DIMENSION
        new_width: int = int(ORIGINAL_WIDTH * (MAX_DIMENSION / ORIGINAL_HEIGHT))

    RESIZED_UNTOUCHED_IMAGE: Final[np.ndarray] = cv.resize(
        UNTOUCHED_IMAGE, (new_width, new_height)
    )

    GRAY_RESIZED_UNTOUCHED_IMAGE: Final[np.ndarray] = cv.cvtColor(
        RESIZED_UNTOUCHED_IMAGE, cv.COLOR_BGR2GRAY
    )

    THRESHOLDED_UNTOUCHED_IMAGE: Final[np.ndarray] = cv.threshold(
        GRAY_RESIZED_UNTOUCHED_IMAGE, 128, 255, cv.THRESH_BINARY
    )[1]

    WHITE_RECTANGLE_CONTOURS: Final[Tuple[int, ...]] = cv.findContours(
        THRESHOLDED_UNTOUCHED_IMAGE,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0, 0),
    )[0]
    if len(WHITE_RECTANGLE_CONTOURS) > 0:
        WHITE_RECTANGLE_LARGEST_CONTOUR: Final[np.ndarray] = max(
            WHITE_RECTANGLE_CONTOURS, key=cv.contourArea
        )
        (
            x_coord,
            y_coord,
            width,
            height,
        ) = cast(tuple[int, ...], cv.boundingRect(WHITE_RECTANGLE_LARGEST_CONTOUR))
        WHITE_RECT_X_COORD: Final[int] = x_coord
        WHITE_RECT_Y_COORD: Final[int] = y_coord
        WHITE_RECT_WIDTH: Final[int] = width
        WHITE_RECT_HEIGHT: Final[int] = height

        angle: int = cv.minAreaRect(WHITE_RECTANGLE_LARGEST_CONTOUR)[-1]

        # If the angle is negative, adjust it to be in the range [0, 90]
        if angle < -45:
            angle += 90

        # Create a rotation matrix
        ROTATION_MATRIX: Final[np.ndarray] = cv.getRotationMatrix2D(
            (
                WHITE_RECT_X_COORD + WHITE_RECT_WIDTH / 2,
                WHITE_RECT_Y_COORD + WHITE_RECT_HEIGHT / 2,
            ),
            angle,
            1,
        )

        # Apply the rotation to the image
        ROTATED_IMAGE: Final[np.ndarray] = cv.warpAffine(
            RESIZED_UNTOUCHED_IMAGE,
            ROTATION_MATRIX,
            (RESIZED_UNTOUCHED_IMAGE.shape[1], RESIZED_UNTOUCHED_IMAGE.shape[0]),
            flags=cv.INTER_LINEAR,
        )
        GRAY_ROTATED_IMAGE: Final[np.ndarray] = cv.cvtColor(
            ROTATED_IMAGE, cv.COLOR_BGR2GRAY
        )

        THRESHOLDED_ROTATED_IMAGE: Final[np.ndarray] = cv.threshold(
            GRAY_ROTATED_IMAGE, 128, 255, cv.THRESH_BINARY
        )[1]
        WHITE_RECTANGLE_CONTOURS_ROTATED_IMAGE: Final[np.ndarray] = cv.findContours(
            THRESHOLDED_ROTATED_IMAGE,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
            offset=(0, 0),
        )[0]
        if len(WHITE_RECTANGLE_CONTOURS_ROTATED_IMAGE) > 0:
            WHITE_RECTANGLE_LARGEST_CONTOUR_ROTATED_IMAGE: Final[np.ndarray] = max(
                WHITE_RECTANGLE_CONTOURS_ROTATED_IMAGE, key=cv.contourArea
            )
            cv.drawContours(
                ROTATED_IMAGE,
                WHITE_RECTANGLE_CONTOURS_ROTATED_IMAGE,
                -1,
                (0, 0, 255),  # BGR : Red
                2,
            )
            (x_coord, y_coord, width, height) = cast(
                tuple[int, ...],
                cv.boundingRect(WHITE_RECTANGLE_LARGEST_CONTOUR_ROTATED_IMAGE),
            )
            WHITE_RECT_X_COORD_ROTATED_IMAGE: Final[int] = x_coord
            WHITE_RECT_Y_COORD_ROTATED_IMAGE: Final[int] = y_coord
            WHITE_RECT_WIDTH_ROTATED_IMAGE: Final[int] = width
            WHITE_RECT_HEIGHT_ROTATED_IMAGE: Final[int] = height
        WHITE_RECT_IMAGE = ROTATED_IMAGE[
            WHITE_RECT_Y_COORD_ROTATED_IMAGE : WHITE_RECT_Y_COORD_ROTATED_IMAGE
            + WHITE_RECT_HEIGHT_ROTATED_IMAGE,
            WHITE_RECT_X_COORD_ROTATED_IMAGE : WHITE_RECT_X_COORD_ROTATED_IMAGE
            + WHITE_RECT_WIDTH_ROTATED_IMAGE,
        ]

        GRAY_WHITE_RECTANGLE: Final[np.ndarray] = cv.cvtColor(
            WHITE_RECT_IMAGE, cv.COLOR_BGR2GRAY
        )
        BLURRED_GRAY_WHITE_RECTANGLE: Final[np.ndarray] = cv.GaussianBlur(
            GRAY_WHITE_RECTANGLE, (9, 9), 2
        )

        RAW_CIRCLES_IN_WHITE_RECTANGLE: Final[np.ndarray] = cv.HoughCircles(
            BLURRED_GRAY_WHITE_RECTANGLE,
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
                key_pressed: int = cv.waitKeyEx(0)
                if key_pressed == 27:  # 27 is 'Esc' key
                    break
                if cv.getWindowProperty(TITLE_WINDOW, cv.WND_PROP_VISIBLE) < 1:
                    break

        else:
            print("No circles detected in the image.")
            cv.imshow("No Circles", GRAY_RESIZED_UNTOUCHED_IMAGE)
            key_pressed = cv.waitKeyEx(0)

    else:
        print("No white rectangle found in the image.")
        cv.imshow("No Light Rectangle", RESIZED_UNTOUCHED_IMAGE)
        key_pressed = cv.waitKeyEx(0)

    cv.destroyAllWindows()
    print("EOP")
except Exception as e:
    print(e)
