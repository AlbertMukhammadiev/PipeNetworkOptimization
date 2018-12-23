import pytest

@pytest.fixture
def network():
    from network import Network, FileNames, ReaderCSV

    FNAME_NODES = '../initial_layouts/square/nodes.csv'
    FNAME_EDGES = '../initial_layouts/square/edges.csv'
    FNAME_COSTS = '../initial_layouts/square/cost_data.csv'
    fnames = FileNames(
        nodes=FNAME_NODES,
        edges=FNAME_EDGES,
        costs=FNAME_COSTS,
    )

    reader = ReaderCSV(fnames)
    return Network(reader)


def test_total_cost(network):
    individual = [
        0,1,0,0,     # 1
        0,0,0,0,     # 2
        0,0,1,1,     # 3
        0,0,1,0,     # 4
        0,0,0,1,     # 5
        0,0,0,0,     # 6
        1,1,0,0,     # 7
        0,1,0,1,     # 8
        0,0,0,0,     # 9
        1,0,1,0,     # 10
        0,0,1,1,     # 11
        0,0,1,0]     # 12
    assert network.total_cost(individual) == 907700
