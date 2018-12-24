import pytest

@pytest.fixture
def network():
    from network import Network
    from data_context import DataContext

    path = '../projects/square_layout/'
    data_context = DataContext(path)
    return Network(data_context)


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
