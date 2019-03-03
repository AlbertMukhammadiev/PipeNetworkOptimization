import pytest
import math
import random


def test_cost_model_mapping(empty_ga_network, cost_model):
    empty_ga_network.cost_model = cost_model
    mapped_model = empty_ga_network._bits_mapping
    assert len(mapped_model) == 16
    bits, pipe_props = random.choice(list(mapped_model.items()))
    assert len(bits) == 4

    sliced_model = cost_model[:8]
    empty_ga_network.cost_model = sliced_model
    mapped_model = empty_ga_network._bits_mapping
    assert len(mapped_model) == len(sliced_model)
    bits, pipe_props = random.choice(list(mapped_model.items()))
    assert len(bits) == 3


def test_cost_model_has0(empty_ga_network):
    model = []
    empty_ga_network.cost_model = model
    diameters = [props.diameter for props in empty_ga_network.cost_model]
    assert 0 in diameters


def test_preliminary_cost(simple_project):
    simple_project.bit_representation = [0, 0, 0, 0] * 12
    assert simple_project.preliminary_cost() == 0
    simple_project.bit_representation = [1, 1, 1, 1] * 12
    assert simple_project.preliminary_cost() == 6000
    simple_project.bit_representation = [
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0,
        0, 0, 1, 1,
        0, 1, 0, 0,
        0, 1, 0, 1,
        0, 1, 1, 0,
        0, 1, 1, 1,
        1, 0, 0, 0,
        1, 0, 0, 1,
        1, 0, 1, 0,
        1, 0, 1, 1,
    ]
    assert simple_project.preliminary_cost() == 2015


def test_total_cost(empty_ga_network, square_layout, cost_model):
    assert empty_ga_network.total_cost() == 0
    empty_ga_network.change_layout(square_layout)
    n_edges = len(square_layout.edges)
    empty_ga_network.cost_model = cost_model

    test_design = ['0000' for _ in range(n_edges)]
    empty_ga_network.bit_representation = test_design
    assert empty_ga_network.total_cost() == 0

    for i in range(0, len(test_design), 3):
        test_design[i] = '0001'
    empty_ga_network.bit_representation = test_design
    assert empty_ga_network.total_cost() == 1380

    test_design = ['0001' for _ in range(n_edges)]
    empty_ga_network.bit_representation = test_design
    assert empty_ga_network.total_cost() == 4140


def test_penalty_cost(simple_project):
    simple_project.bit_representation = [0, 0, 0, 0] * 2 + [1, 1, 1, 1] * 10
    assert simple_project.total_cost() == simple_project._max_possible_cost

    simple_project.bit_representation = [0, 0, 0, 0] * 7 + [1, 1, 1, 1] * 5
    assert simple_project.total_cost() == simple_project._max_possible_cost * 4


def test_max_possible_cost(empty_ga_network, square_layout, cost_model):
    empty_ga_network.change_layout(square_layout)
    assert empty_ga_network._max_possible_cost == 0
    empty_ga_network.cost_model = cost_model
    assert empty_ga_network._max_possible_cost == 90000
