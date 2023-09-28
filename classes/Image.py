import cv2 as cv


class Image:
    def __init__(self, path, max_dimension, title) -> None:
        self.max_dimension = max_dimension
        self.path = path
        self.title = title
        self.untouched_image = None
        self.original_width = None
        self.original_height = None
        self.resized_image = None
        self.gray_resized_image = None
        self.thresholded_image = None

    @property
    def path(self):
        return self._path

    @property
    def max_dimension(self):
        return self._max_dimension

    @property
    def title(self):
        return self._title

    @property
    def untouched_image(self):
        return self._read_image

    @property
    def original_width(self):
        return self._original_width

    @property
    def original_height(self):
        return self._original_height

    @property
    def resized_image(self):
        return self._resized_image

    @property
    def gray_resized_image(self):
        return self._gray_resized_image

    @property
    def thresholded_image(self):
        return self._thresholded_image

    @untouched_image.setter
    def untouched_image(self, untouched_image):
        self._untouched_image = untouched_image

    @resized_image.setter
    def resized_image(self, new_resized_image):
        self._resized_image = new_resized_image

    @gray_resized_image.setter
    def gray_resized_image(self, new_gray_resized):
        self._gray_resized_image = new_gray_resized

    @thresholded_image.setter
    def thresholded_image(self, new_thresholded_image):
        self._thresholded_image = new_thresholded_image

    def read(self):
        self.untouched_image = cv.imread(self.path())

    def save_dimension(self, round_number: int):
        (
            self.original_height,
            self.original_width,
        ) = self.untouched_image().shape[:round_number]

    def resize(self):
        if self.original_width() > self.original_height():
            new_width = self.max_dimension
            new_height = int(
                self.original_height * (self.max_dimension / self.original_width)
            )
        else:
            new_height = self.max_dimension
            new_width = int(
                self.original_width * (self.max_dimension / self.original_height)
            )
        self.resized_image = cv.resize(self.untouched_image(), (new_width, new_height))

    def grayscale(self):
        self.gray_resized_image = cv.cvtColor(self.resized_image, cv.COLOR_BGR2GRAY)

    def threshold(self):
        self.thresholded_image = cv.threshold(
            self.gray_resized_image, 128, 255, cv.THRESH_BINARY
        )[1]
