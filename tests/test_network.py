import pytest
import math
import random


def test_development_cost(simple_project):
    simple_project.chromosome = [
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
        1, 0, 1, 1,]
    assert simple_project.preliminary_cost() == 2015

    simple_project.chromosome = [
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
    ]
    assert simple_project.preliminary_cost() == 384


def test_preliminary_cost(simple_project):
    simple_project.bit_repr = [0, 0, 0, 0] * 12
    assert simple_project.preliminary_cost() == 0
    simple_project.bit_repr = [1, 1, 1, 1] * 12
    assert simple_project.preliminary_cost() == 6000
    simple_project.bit_repr = [
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
    empty_ga_network.bit_repr = test_design
    assert empty_ga_network.total_cost() == 0

    for i in range(0, len(test_design), 3):
        test_design[i] = '0001'
    empty_ga_network.bit_repr = test_design
    assert empty_ga_network.total_cost() == 1380

    test_design = ['0001' for _ in range(n_edges)]
    empty_ga_network.bit_repr = test_design
    assert empty_ga_network.total_cost() == 4140


def test_penalty_cost(simple_project):
    simple_project.bit_repr = [0, 0, 0, 0] * 2 + [1, 1, 1, 1] * 10
    assert simple_project.total_cost() == simple_project._max_possible_cost

    simple_project.bit_repr = [0, 0, 0, 0] * 7 + [1, 1, 1, 1] * 5
    assert simple_project.total_cost() == simple_project._max_possible_cost * 4


def test_max_possible_cost(empty_ga_network, square_layout, cost_model):
    empty_ga_network.change_layout(square_layout)
    assert empty_ga_network._max_possible_cost == 0
    empty_ga_network.cost_model = cost_model
    assert empty_ga_network._max_possible_cost == 90000
