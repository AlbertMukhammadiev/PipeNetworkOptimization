import pytest
import layouts


def test_1x1_layout():
    layout = layouts.SquareLayout(1, 1)
    assert len(layout.edges) == 0
    layout = layouts.HexagonLayout(1, 1)
    assert len(layout.edges) == 0


def test_3x3_layout():
    layout = layouts.SquareLayout(3, 3)
    assert len(layout.edges) == 12
    layout = layouts.HexagonLayout(3, 3)
    assert len(layout.edges) == 8
