import pytest
import math
import random


def test_cost_model_mapping(empty_ga_network, cost_model):
    empty_ga_network.cost_model = cost_model
    mapped_model = empty_ga_network.cost_model
    assert len(mapped_model) == 16
    bits, pipe_props = random.choice(list(mapped_model.items()))
    assert len(bits) == 4

    sliced_model = cost_model[:8]
    empty_ga_network.cost_model = sliced_model
    mapped_model = empty_ga_network.cost_model
    assert len(mapped_model) == len(sliced_model)
    bits, pipe_props = random.choice(list(mapped_model.items()))
    assert len(bits) == 3


def test_cost_model_has0(empty_ga_network):
    model = []
    empty_ga_network.cost_model = model
    model_values = list(empty_ga_network.cost_model.values())
    diameters = [props.diameter for props in model_values]
    assert 0 in diameters


def test_total_cost(empty_ga_network, square_layout, cost_model):
    assert empty_ga_network.total_cost() == 0
    empty_ga_network.layout = square_layout
    n_edges = len(square_layout.edges)
    empty_ga_network.cost_model = cost_model

    test_design = ['0000' for _ in range(n_edges)]
    empty_ga_network.design = test_design
    assert empty_ga_network.total_cost() == 0

    for i in range(0, len(test_design), 3):
        test_design[i] = '0001'
    empty_ga_network.design = test_design
    assert empty_ga_network.total_cost() == 1380

    test_design = ['0001' for _ in range(n_edges)]
    empty_ga_network.design = test_design
    assert empty_ga_network.total_cost() == 4140

    assert empty_ga_network._max_cost() == 90000
