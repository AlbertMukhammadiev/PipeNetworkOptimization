import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

from properties import *


class Network:
    def __init__(self, sinks, sources):
        self._design = nx.Graph()
        self._init_task_entities(sinks, sources)
        self._init_auxiliary_entities()
        self._init_design_constants()

    def _init_task_entities(self, sinks, sources):
        self._sources = sources
        self._sinks = sinks
        self._cost_model = list()

    def _init_auxiliary_entities(self):
        self._edges_props = dict()
        self._nodes_props = dict()
        self._drawing_kwargs = {
            'node_color': 'aqua',
            'node_size': 100,
            'alpha': 0.8,
            'font_size': 1,
            'rotate': False,
            'labels': self._nodes_props,
            # 'edge_labels': self._edges_props,
        }

    def _init_design_constants(self):
        self._max_possible_cost = 0
        self.VELOCITY = 1

    @staticmethod
    def flow_rate(diameter: float, velocity: float):
        # TODO the dependence on units of measurement
        return 1 / 4 * math.pi * diameter ** 2 * velocity / 1000

    @property
    def cost_model(self):
        return self._cost_model.copy()

    @cost_model.setter
    def cost_model(self, value):
        self._cost_model.clear()
        if 0 not in [props.diameter for props in value]:
            value.append(PipeProps(diameter=0, cost=0))
        self._cost_model.extend(value)
        self._update_max_possible_cost()

    def draw(self):
        pos = {node: self._nodes_props[node].position for node in self._design.nodes}
        nx.draw_networkx_nodes(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_edges(self._design, pos, **self._drawing_kwargs)
        plt.show()

    def draw_in_detail(self):
        pos = {node: self._nodes_props[node].position for node in self._design.nodes}
        nx.draw_networkx_nodes(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_labels(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_edges(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_edge_labels(self._design, pos, **self._drawing_kwargs)
        plt.show()

    def draw_pdf(self, path):
        pos = {node: self._nodes_props[node].position for node in self._design.nodes}
        nx.draw_networkx_nodes(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_labels(self._design, pos, **self._drawing_kwargs)
        nx.draw_networkx_edges(self._design, pos, **self._drawing_kwargs)
        edge_labels = {edge: props for edge, props in self._edges_props.items() if edge in self._design.edges}
        nx.draw_networkx_edge_labels(self._design, pos, edge_labels=edge_labels, **self._drawing_kwargs)
        plt.savefig(path, format='pdf')
        plt.close()

    def total_cost(self):
        return self.preliminary_cost() + self.penalty_cost()

    def preliminary_cost(self):
        costs = np.array([self._edges_props[edge].cost for edge in self._design.edges])
        lengths = np.array([self._edges_props[edge].length for edge in self._design.edges])
        return sum(costs * lengths)

    def penalty_cost(self):
        costs = np.array([self._edges_props[edge].cost for edge in self._design.edges])
        lengths = np.array([self._edges_props[edge].length for edge in self._design.edges])
        rates = []
        for edge in self._design.edges:
            props = self._edges_props[edge]
            rates.append(math.floor(props.actual_flow / props.flow_rate))
        return sum(costs * lengths * np.array(rates))

    def _add_node(self, node, position, demand=0):
        self._nodes_props[node] = NodeProps(
            position=position,
            demand=demand
        )

    def _add_edge(self, u, v,
                  length=1, diameter=1, flow_rate=0,
                  actual_flow=0, No=None, cost=0):
        props = EdgeProps(
            length=length,
            diameter=diameter,
            flow_rate=flow_rate,
            actual_flow=actual_flow,
            No=No,
            cost=cost,
        )
        self._edges_props[u, v] = props
        self._edges_props[v, u] = props
        self._design.add_edge(u, v)

    def _update_node(self, node, position, demand):
        self._nodes_props[node].position = position
        self._nodes_props[node].demand = demand

    def _update_edge(self, u, v, diameter, cost):
        self._edges_props[u, v].actual_flow = 0
        self._edges_props[u, v].cost = cost
        self._edges_props[u, v].diameter = diameter
        self._edges_props[u, v].flow_rate =\
            Network.flow_rate(diameter, self.VELOCITY)

    def _update_max_possible_cost(self):
        most_expensive = max([pipe.cost for pipe in self.cost_model])
        weights = [props.length * most_expensive for props in self._edges_props.values()]
        self._max_possible_cost = sum(weights) / 2

    def _nearest_node(self, point):
        nearest = None
        dist_nearest = 1000000
        for node in self._design.nodes:
            x_node, y_node = self._nodes_props[node].position
            current_dist = math.hypot(y_node - point[1], x_node - point[0])
            if current_dist < dist_nearest:
                nearest = node
                dist_nearest = current_dist
        return nearest


class NetworkGA(Network):
    def __init__(self, sinks, sources):
        super().__init__(sinks, sources)

        self._nbits_4pipe = 0
        self._bits_mapping = dict()
        self._bit_representation = ''
        self._edge_by_No = dict()

    @Network.cost_model.setter
    def cost_model(self, value):
        # TODO call setter of base class
        self._cost_model.clear()
        if 0 not in [props.diameter for props in value]:
            value.append(PipeProps(diameter=0, cost=0))
        self._cost_model.extend(value)
        self._update_max_possible_cost()
        ##################################################################

        self._bits_mapping.clear()
        key_len = (len(bin(len(self.cost_model) - 1)) - 2)
        sorted_props = sorted(self.cost_model, key=lambda pipe: pipe.diameter)
        for number in range(2 ** key_len):
            key = bin(number)[2:].zfill(key_len)
            try:
                self._bits_mapping[key] = sorted_props[number]
            except IndexError:
                self._bits_mapping[key] = sorted_props[-1]

    @property
    def bit_representation(self):
        return self._bit_representation

    @bit_representation.setter
    def bit_representation(self, value):
        self._reset()
        self._bit_representation = ''.join(str(bit) for bit in value)
        self._redesign()

    def total_cost(self) -> float:
        n_isolated_sources = self.calculate_flows()
        if n_isolated_sources > 0:
            return self._max_possible_cost * n_isolated_sources
        else:
            return super().total_cost()

    def change_layout(self, layout):
        numeration = layout.numerated_edges
        for edge in layout.edges:
            self._add_edge(*edge, No=numeration[edge.v, edge.u])
            self._add_node(edge.u, position=edge.u)
            self._add_node(edge.v, position=edge.v)
            self._edge_by_No[numeration[edge]] = edge

        for position, demand in self._sinks.items():
            nearest = self._nearest_node(position)
            self._update_node(nearest, position=position, demand=demand)
        for position, demand in self._sources.items():
            nearest = self._nearest_node(position)
            self._update_node(nearest, position=position, demand=demand)

    def calculate_flows(self):
        n_isolated_sources = 0
        for source, demand in self._sources.items():
            for sink, _ in self._sinks.items():
                try:
                    traces = nx.shortest_path(self._design, source, sink)
                    for u, v in zip(traces, traces[1:]):
                        self._edges_props[u, v].actual_flow += demand
                except nx.exception.NetworkXNoPath:
                    n_isolated_sources += 1
        return n_isolated_sources

    def nbits_required(self):
        return (len(bin(len(self.cost_model) - 1)) - 2) * len(self._edges_props) // 2

    def _reset(self):
        edges = list(self._design.edges())
        self._design.remove_edges_from(edges)

    def _redesign(self):
        substring_len = (len(bin(len(self.cost_model) - 1)) - 2)
        bits = self.bit_representation
        substrings = [bits[i:i + substring_len] for i in range(0, len(bits), substring_len)]
        for i, substring in enumerate(substrings):
            pipe_props = self._bits_mapping[substring]
            if pipe_props.cost:
                edge = self._edge_by_No[i]
                self._design.add_edge(*edge)
                self._update_edge(
                    *edge,
                    diameter=pipe_props.diameter,
                    cost=pipe_props.diameter,
                )


if __name__ == '__main__':
    pass
