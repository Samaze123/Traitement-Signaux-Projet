import numpy as np


class Rectangle:
    def __init__(self) -> None:
        self.contours: tuple[np.ndarray, ...] = None
        self.largest_contour: np.ndarray = None
        self.coord: list[int] = [0, 0]
        self.rotation_matrix: np.ndarray = None
        self.size: list[int] = [0, 0]
        self.ratio: float = 0.0

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, new_ratio: list[int]):
        self._ratio = new_ratio

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size: list[int]):
        self._size = new_size

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, new_rotation_matrix: np.ndarray):
        self._rotation_matrix = new_rotation_matrix

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, new_coord: list[int]):
        self._coord = new_coord

    @property
    def largest_contour(self):
        return self._largest_contour

    @largest_contour.setter
    def largest_contour(self, new_largest_contour: np.ndarray):
        self._largest_contour = new_largest_contour

    @property
    def contours(self):
        return self._contours

    @contours.setter
    def contours(self, new_contours: tuple[np.ndarray, ...]):
        self._contours = new_contours
