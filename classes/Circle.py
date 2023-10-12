class Circle:
    def __init__(self) -> None:
        self.selected: bool = False
        self.coord: object = {"x": 0, "y": 0}
        self.radius: int = 0

    @property
    def radius(self) -> int:
        return self._radius

    @radius.setter
    def radius(self, new_radius: int) -> None:
        self._radius = new_radius

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, new_selected: bool) -> None:
        self._selected = new_selected

    @property
    def coord(self) -> object:
        return self._coord

    @coord.setter
    def coord(self, new_coord: object) -> None:
        self._coord = new_coord
