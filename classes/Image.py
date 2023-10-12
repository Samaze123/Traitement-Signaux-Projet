import cv2 as cv
import numpy as np
from classes.Rectangle import Rectangle


class Image:
    def __init__(self) -> None:
        self.original_image: np.ndarray = []
        self.gray_image: np.ndarray = []
        self.thresholded_image: np.ndarray = []
        self.resized_image: np.ndarray = []
        self.blurred_image: np.ndarray = []
        self.original_dimension: list[int] = [0, 0]
        self.max_dimension: int = 0

    @property
    def blurred_image(self) -> np.ndarray:
        return self._blurred_image

    @blurred_image.setter
    def blurred_image(self, new_blurred_image: np.ndarray) -> None:
        self._blurred_image = new_blurred_image

    @property
    def resized_image(self) -> np.ndarray:
        return self._resized_image

    @resized_image.setter
    def resized_image(self, new_resized_image: np.ndarray) -> None:
        self._resized_image = new_resized_image

    @property
    def max_dimension(self) -> int:
        return self._max_dimension

    @max_dimension.setter
    def max_dimension(self, new_max_dimension) -> None:
        self._max_dimension = new_max_dimension

    @property
    def original_dimension(self) -> list[int]:
        return self._original_dimension

    @original_dimension.setter
    def original_dimension(self, new_original_dimension) -> None:
        self._original_dimension = new_original_dimension

    @property
    def thresholded_image(self) -> np.ndarray:
        return self._thresholded_image

    @thresholded_image.setter
    def thresholded_image(self, new_thresholded_image: np.ndarray) -> None:
        self._thresholded_image = new_thresholded_image

    @property
    def gray_image(self) -> np.ndarray:
        return self._gray_image

    @gray_image.setter
    def gray_image(self, new_gray_image: np.ndarray) -> None:
        self._gray_image = new_gray_image

    @property
    def original_image(self) -> np.ndarray:
        return self._original_image

    @original_image.setter
    def original_image(self, new_original_image: np.ndarray) -> None:
        self._original_image = new_original_image

    def findRotation(self, rectangle: Rectangle) -> np.ndarray:
        """
        Calculates the rotation matrix for a given rectangle and applies the rotation to an image.

        Args:
            self: The current instance of the class.
            rectangle (Rectangle): The rectangle object to be rotated.

        Returns:
            np.ndarray: The rotated image.

        Example:
            ```python
            image = Image()
            rectangle = Rectangle()
            rotated_image = image.findRotation(rectangle)
            cv.imshow("Rotated Image", rotated_image)
            cv.waitKey(0)
            ```
        """

        angle: int = cv.minAreaRect(rectangle.largest_contour)[-1]

        # If the angle is negative, adjust it to be in the range [0, 90]
        if angle < -45:
            angle += 90

        # Create a rotation matrix
        rectangle.rotation_matrix = cv.getRotationMatrix2D(
            (
                rectangle.coord[0] + rectangle.size[0] / 2,
                rectangle.coord[1] + rectangle.size[1] / 2,
            ),
            angle,
            1,
        )
        # Apply the rotation to the image
        return cv.warpAffine(
            self.resized_image,
            rectangle.rotation_matrix,
            (
                self.resized_image.shape[1],
                self.resized_image.shape[0],
            ),
            flags=cv.INTER_LINEAR,
        )
