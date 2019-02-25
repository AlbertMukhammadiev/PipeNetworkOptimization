import pytest
from layouts import SquareLayout, HexagonLayout
from network import Network, NetworkGA
from properties import PipeProps


@pytest.fixture(scope='session')
def square_layout():
    return SquareLayout(10, 10)


@pytest.fixture(scope='session')
def hexagon_layout():
    return HexagonLayout(10, 10)


@pytest.fixture(scope='session')
def empty_network():
    return Network(dict(), dict())


@pytest.fixture(scope='session')
def empty_ga_network():
    return NetworkGA(dict(), dict())


@pytest.fixture(scope='session')
def cost_model():
    return [
        PipeProps(diameter=0.0, cost=0.0), PipeProps(diameter=80.0, cost=23.0),
        PipeProps(diameter=100.0, cost=32.0), PipeProps(diameter=120.0, cost=50.0),
        PipeProps(diameter=140.0, cost=60.0), PipeProps(diameter=160.0, cost=90.0),
        PipeProps(diameter=180.0, cost=130.0), PipeProps(diameter=200.0, cost=170.0),
        PipeProps(diameter=220.0, cost=300.0), PipeProps(diameter=240.0, cost=340.0),
        PipeProps(diameter=260.0, cost=390.0), PipeProps(diameter=280.0, cost=430.0),
        PipeProps(diameter=300.0, cost=470.0), PipeProps(diameter=320.0, cost=500.0),
    ]



