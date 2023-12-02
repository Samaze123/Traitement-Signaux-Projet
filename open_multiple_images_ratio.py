from typing import Final, Tuple, cast
import traceback
import cv2 as cv
import numpy as np
import os

from classes.Image import Image
from classes.Rectangle import Rectangle
from classes.Colors import Colors

try:
    # ================#
    # GLOBAL VARIABLE #
    # ================#
    MAX_DIMENSION: Final[int] = 800
    MARGIN_ERROR: Final[float] = 1.2
    FOLDER_PATH: Final[str] = "./images/images_converted"
    TITLE_WINDOW: Final[str] = "Detected Circles"

    global_selected_circle: Tuple[int] = None
    global_white_rectangle_ratio: Tuple[int] = None
    global_selected_filename = None

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
                    print_selected_circle_info(circle, white_rectangle)
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

    def calculate_white_rectangle_position(
        circle: np.ndarray, rectangle: Rectangle
    ) -> Tuple[float, float]:
        x, y, r = circle
        CIRCLE_DIAMETER_PIXEL: Final[int] = r * 2

        CIRCLE_DISTANCE_FROM_TOP_PIXEL: Final[int] = y - rectangle.coord[1]
        CIRCLE_DISTANCE_FROM_LEFT_PIXEL: Final[int] = x - rectangle.coord[0]

        return (
            CIRCLE_DIAMETER_PIXEL,
            CIRCLE_DISTANCE_FROM_TOP_PIXEL,
            CIRCLE_DISTANCE_FROM_LEFT_PIXEL,
        )

    def print_selected_circle_info(
        CIRCLE_DIAMETER_PIXEL: int,
        CIRCLE_DISTANCE_FROM_TOP_PIXEL: int,
        CIRCLE_DISTANCE_FROM_LEFT_PIXEL: int,
    ) -> None:
        """
        Prints information about a selected circle and rectangle.

        Args:
            circle (np.ndarray): The selected circle represented as an array.
            rectangle (Rectangle): The selected rectangle.

        Returns:
            Tuple[float, float]: A tuple containing the width and height of the selected rectangle in centimeters.

        Examples:
            ```python
            circle = np.array([x, y, radius])
            rectangle = Rectangle(width, height, (x, y))
            print_selected_circle_info(circle, rectangle)
            ```
        """

        print(f"{Colors.PURPLE}####################")
        print(
            f"\n=====\nSelected Circle Diameter (in pixels): {CIRCLE_DIAMETER_PIXEL}\n-----\n"
        )

        print(f"Distance from top (in pixels): {CIRCLE_DISTANCE_FROM_TOP_PIXEL}")
        print(
            f"Distance from left (in pixels): {CIRCLE_DISTANCE_FROM_LEFT_PIXEL}{Colors.RESET}"
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

    def get_circles(image: Image) -> np.ndarray:
        RAW_CIRCLES_IN_WHITE_RECTANGLE: Final[np.ndarray] = cv.HoughCircles(
            image.blurred_image,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # Minimum distance between detected circles
            param1=255,  # Upper threshold for edge detection
            param2=13,  # Threshold for circle detection
            minRadius=1,  # Minimum circle radius
            maxRadius=50,  # Maximum circle radius (adjust as needed)
        )

        if RAW_CIRCLES_IN_WHITE_RECTANGLE is not None:
            return np.round(RAW_CIRCLES_IN_WHITE_RECTANGLE[0, :]).astype("int")
        return None

    # ========#
    # PROGRAM #
    # ========#
    for i, filename in enumerate(os.listdir(FOLDER_PATH)):
        if (
            filename.endswith(".jpeg")
            or filename.endswith(".jpg")
            or filename.endswith(".png")
        ):
            IMAGE_PATH: Final[str] = os.path.join(FOLDER_PATH, filename)
            opened_image: Image = Image()
            white_rectangle: Rectangle = Rectangle()
            rotated_image: Image = Image()
            rotated_white_rectangle: Rectangle = Rectangle()
            white_rectangle_image = Image()
            opened_image.max_dimension = MAX_DIMENSION
            opened_image.original_image = cv.imread(IMAGE_PATH)
            opened_image.original_dimension = opened_image.original_image.shape[
                :2
            ]  #  h x w

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

            opened_image.gray_image = cv.cvtColor(
                opened_image.resized_image, cv.COLOR_BGR2GRAY
            )

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
                ) = cast(
                    tuple[int, ...], cv.boundingRect(white_rectangle.largest_contour)
                )
                rotated_image.original_image = opened_image.findRotation(
                    white_rectangle
                )

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

                CIRCLES_IN_WHITE_RECTANGLE: Final[np.ndarray] = get_circles(
                    white_rectangle_image
                )

                if CIRCLES_IN_WHITE_RECTANGLE is None:
                    print(
                        f"{Colors.RED}Pas de cercles détectés sur l'image {filename}{Colors.RESET}"
                    )
                elif len(CIRCLES_IN_WHITE_RECTANGLE) == 1:
                    for j, (
                        CIRCLE_X_COORD,
                        CIRCLE_Y_CCORD,
                        CIRCLE_RADIUS,
                    ) in enumerate(CIRCLES_IN_WHITE_RECTANGLE):
                        if j == 0:
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
                    (
                        diameter_pixel,
                        circle_from_top,
                        circle_from_left,
                    ) = calculate_white_rectangle_position(
                        CIRCLES_IN_WHITE_RECTANGLE[0], white_rectangle
                    )

                    WHITE_RECTANGLE_RATIO: Final[float] = (
                        white_rectangle.size[0] / white_rectangle.size[1]
                        if white_rectangle.size[0] > white_rectangle.size[1]
                        else white_rectangle.size[1] / white_rectangle.size[0]
                        if white_rectangle.size[1] > white_rectangle.size[0]
                        else 1.0
                    )

                    if i == 0:
                        global_selected_filename = filename
                        global_white_rectangle_ratio = WHITE_RECTANGLE_RATIO
                        print(
                            f"{Colors.YELLOW}##########################\nFirst Rectangle : {global_white_rectangle_ratio}\n{filename}{Colors.RESET}"
                        )
                        # print_informations_circles(CIRCLES_IN_WHITE_RECTANGLE)
                        cv.imshow(TITLE_WINDOW, white_rectangle_image.original_image)
                        # cv.setMouseCallback(TITLE_WINDOW, select_circle)
                        while True:
                            key_pressed: int = cv.waitKeyEx(0)
                            if key_pressed == 27:  # 27 is 'Esc' key
                                break
                            if (
                                cv.getWindowProperty(TITLE_WINDOW, cv.WND_PROP_VISIBLE)
                                < 1
                            ):
                                break
                    else:
                        print(
                            f"{Colors.YELLOW}##########################\n{WHITE_RECTANGLE_RATIO}{Colors.RESET}"
                        )
                        if (
                            WHITE_RECTANGLE_RATIO
                            > MARGIN_ERROR * global_white_rectangle_ratio
                        ):
                            print(
                                f"{Colors.ORANGE}La planche sur l'image {filename} est trop grande.{Colors.RESET}"
                            )
                        elif (
                            WHITE_RECTANGLE_RATIO
                            < global_white_rectangle_ratio / MARGIN_ERROR
                        ):
                            print(
                                f"{Colors.BLUE}La planche sur l'image {filename} est trop petite.{Colors.RESET}"
                            )
                        else:
                            print(
                                f"{Colors.GREEN}La planche sur l'image {filename} est identique.{Colors.RESET}"
                            )
                elif len(CIRCLES_IN_WHITE_RECTANGLE) > 1:
                    print(
                        f"{Colors.RED}Plusieurs cercles détectés sur l'image {filename}{Colors.RESET}"
                    )
                    if i == 1:
                        print(
                            f"{Colors.RED}Problème avec la première image{Colors.RESET}"
                        )
                        break
            else:
                print(
                    f"{Colors.RED}Pas de rectangle détectés sur l'image {filename}{Colors.RESET}"
                )
        else:
            print(
                f"{Colors.RED}Le fichier {filename} ne contient pas la bonne extension{Colors.RESET}"
            )
    print("EOP")
except Exception:
    traceback.print_exc()
