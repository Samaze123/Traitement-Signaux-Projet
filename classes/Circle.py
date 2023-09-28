class Circle:
    def __init__(self, diameter) -> None:
        self._diameter = diameter

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, new_diameter):
        self._diameter = new_diameter
