import pytest
from layouts import Layout
from network import Network, NetworkGA, Point
from properties import PipeProps


@pytest.fixture(scope='function')
def square_layout():
    return SquareLayout(10, 10)


@pytest.fixture(scope='function')
def hexagon_layout():
    return HexagonLayout(10, 10)


@pytest.fixture(scope='function')
def empty_network():
    return Network(list(), list())


@pytest.fixture(scope='function')
def empty_ga_network():
    return NetworkGA(dict(), dict())


@pytest.fixture(scope='session')
def cost_model():
    return [
        PipeProps(diameter=0.0, cost=0.0),
        PipeProps(diameter=80.0, cost=23.0),
        PipeProps(diameter=100.0, cost=32.0),
        PipeProps(diameter=120.0, cost=50.0),
        PipeProps(diameter=140.0, cost=60.0),
        PipeProps(diameter=160.0, cost=90.0),
        PipeProps(diameter=180.0, cost=130.0),
        PipeProps(diameter=200.0, cost=170.0),
        PipeProps(diameter=220.0, cost=300.0),
        PipeProps(diameter=240.0, cost=340.0),
        PipeProps(diameter=260.0, cost=390.0),
        PipeProps(diameter=280.0, cost=430.0),
        PipeProps(diameter=300.0, cost=470.0),
        PipeProps(diameter=320.0, cost=500.0),
    ]


@pytest.fixture(scope='function')
def simple_project(cost_model):
    layout = Layout()
    layout.add_node(node=1, position=Point(x=0, y=0))
    layout.add_node(node=2, position=Point(x=1, y=0))
    layout.add_node(node=3, position=Point(x=0, y=1))
    layout.add_node(node=4, position=Point(x=2, y=0))
    layout.add_node(node=5, position=Point(x=1, y=1))
    layout.add_node(node=6, position=Point(x=0, y=2))
    layout.add_node(node=7, position=Point(x=2, y=1))
    layout.add_node(node=8, position=Point(x=1, y=2))
    layout.add_node(node=9, position=Point(x=2, y=2))

    layout.add_edge(1, 2)
    layout.add_edge(1, 3)
    layout.add_edge(2, 4)
    layout.add_edge(2, 5)
    layout.add_edge(3, 5)
    layout.add_edge(3, 6)
    layout.add_edge(4, 7)
    layout.add_edge(5, 7)
    layout.add_edge(5, 8)
    layout.add_edge(6, 8)
    layout.add_edge(7, 9)
    layout.add_edge(8, 9)

    network = NetworkGA(layout, cost_model)

    network.add_sink(demand=-120, position=Point(x=2, y=2))
    network.add_source(demand=10, position=Point(x=0, y=0))
    network.add_source(demand=20, position=Point(x=1, y=0))
    network.add_source(demand=10, position=Point(x=0, y=1))
    network.add_source(demand=20, position=Point(x=2, y=0))
    network.add_source(demand=10, position=Point(x=1, y=1))
    network.add_source(demand=20, position=Point(x=0, y=2))
    network.add_source(demand=10, position=Point(x=2, y=1))
    network.add_source(demand=20, position=Point(x=1, y=2))
    return network
