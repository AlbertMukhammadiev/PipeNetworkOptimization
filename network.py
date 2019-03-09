import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

from properties import *
from layouts import *


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
        pos, kwargs = self._drawing_configurations()
        nx.draw_networkx_nodes(self._design, pos, **kwargs)
        nx.draw_networkx_edges(self._design, pos, **kwargs)
        plt.show()

    def draw_in_detail(self):
        pos, kwargs = self._drawing_configurations()
        nx.draw_networkx_nodes(self._design, pos, **kwargs)
        nx.draw_networkx_labels(self._design, pos, **kwargs)
        nx.draw_networkx_edges(self._design, pos, **kwargs)
        nx.draw_networkx_edge_labels(self._design, pos, **kwargs)
        plt.show()

    def draw_pdf(self, path):
        pos, kwargs = self._drawing_configurations()
        nx.draw_networkx_nodes(self._design, pos, **kwargs)
        nx.draw_networkx_labels(self._design, pos, **kwargs)
        nx.draw_networkx_edges(self._design, pos, **kwargs)
        nx.draw_networkx_edge_labels(self._design, pos, **kwargs)
        plt.savefig(path, format='pdf')
        plt.close()

    def total_cost(self):
        return self.preliminary_cost() + self.penalty_cost()

    def preliminary_cost(self):
        costs = [self._edges_props[edge]['cost'] for edge in self._design.edges]
        lengths = [self._edges_props[edge]['length'] for edge in self._design.edges]
        return sum(np.array(costs) * np.array(lengths))

    def penalty_cost(self):
        costs = [self._edges_props[edge]['cost'] for edge in self._design.edges]
        lengths = [self._edges_props[edge]['length'] for edge in self._design.edges]
        rates = []
        for edge in self._design.edges:
            props = self._edges_props[edge]
            rates.append(math.floor(props['actual_flow'] / props['flow_rate']))
        return sum(np.array(costs) * np.array(lengths) * np.array(rates))

    def _drawing_configurations(self):
        pos = {node: self._nodes_props[node]['position'] for node in self._design.nodes}
        kwargs = {
            'node_color': [],
            'node_size': 30,
            'alpha': 0.8,
            'font_size': 1,
            'rotate': False,
            'labels': self._nodes_props,
            'edge_labels': {}
        }

        for node in self._design.nodes:
            if self._nodes_props[node]['demand'] > 0:
                kwargs['node_color'].append('aqua')
            elif self._nodes_props[node]['demand'] < 0:
                kwargs['node_color'].append('green')
            else:
                kwargs['node_color'].append('white')
        for edge in self._design.edges:
            props = self._edges_props[edge]
            label = f'd: {props["diameter"]}\nc: {props["cost"]}'
            kwargs['edge_labels'][edge] = label

        return pos, kwargs

    def _add_node(self, node, position, demand=0):
        self._nodes_props[node] = dict(
            position=position,
            demand=demand,
        )

    def _add_edge(self, u, v, index,
                  length=1, diameter=1, cost=0,
                  flow_rate=0, actual_flow=0):
        props = dict(
            length=length,
            diameter=diameter,
            flow_rate=flow_rate,
            actual_flow=actual_flow,
            index=index,
            cost=cost,
        )

        self._edges_props[u, v] = props
        self._edges_props[v, u] = props
        self._design.add_edge(u, v)

    def _update_node(self, node, position, demand):
        self._nodes_props[node]['position'] = position
        self._nodes_props[node]['demand'] = demand

    def _update_edge(self, u, v, diameter, cost, bits):
        self._edges_props[u, v]['actual_flow'] = 0
        self._edges_props[u, v]['cost'] = cost
        self._edges_props[u, v]['diameter'] = diameter
        self._edges_props[u, v]['flow_rate'] =\
            Network.flow_rate(diameter, self.VELOCITY)
        self._edges_props[u, v]['bits'] = bits

    def _update_max_possible_cost(self):
        most_expensive = max([pipe.cost for pipe in self.cost_model])
        lengths = [props['length'] for props in self._edges_props.values()]
        self._max_possible_cost = sum(most_expensive * np.array(lengths)) / 2

    # TODO fix function
    def _nearest_node(self, point):
        nearest = None
        dist_nearest = 1000000
        # for node in self._design.nodes:
        for node, props in self._nodes_props.items():
            # x_node, y_node = self._nodes_props[node]['position']
            x_node, y_node = props['position']
            current_dist = math.hypot(y_node - point[1], x_node - point[0])
            if current_dist < dist_nearest:
                nearest = node
                dist_nearest = current_dist
        return nearest

    def _reset_design(self):
        edges = list(self._design.edges())
        self._design.remove_edges_from(edges)


class NetworkGA(Network):
    def __init__(self, sinks, sources):
        super().__init__(sinks, sources)
        self._layout = nx.Graph()
        self._nbits_4pipe = 0
        self._bits_mapping = dict()
        self._bit_representation = np.array([])
        self._edge_by_index = dict()

    @property
    def cost_model(self):
        return self._cost_model.copy()

    # TODO call setter of base class
    @cost_model.setter
    def cost_model(self, value):
        ##############################33
        self._cost_model.clear()
        if 0 not in [props.diameter for props in value]:
            value.append(PipeProps(diameter=0, cost=0))
        self._cost_model.extend(value)
        self._update_max_possible_cost()
        #####################################

        self._bits_mapping.clear()
        # math.sqrt(1 << (len(self.cost_model) - 1).bit_length())
        key_len = (len(bin(len(self.cost_model) - 1)) - 2)
        sorted_props = sorted(self.cost_model, key=lambda pipe: pipe.diameter)
        for number in range(2 ** key_len):
            key = bin(number)[2:].zfill(key_len)
            try:
                self._bits_mapping[key] = sorted_props[number]
            except IndexError:
                self._bits_mapping[key] = sorted_props[-1]

    @property
    def layout(self):
        return self._layout

    # TODO reset props and indexing?
    @layout.setter
    def layout(self, value):
        self._reset_design()
        self._layout = value
        self._init_edge_by_index()
        # TODO indexing was replaced by _init_edge_by_index
        indexing = {k: v for v, k in self._edge_by_index.items()}
        for edge in self._layout.edges:
            self._add_edge(*edge, index=indexing[edge.v, edge.u])
            self._add_node(edge.u, position=edge.u)
            self._add_node(edge.v, position=edge.v)

        self._place_productive_nodes_on_layout()

    @property
    def bit_representation(self):
        return self._bit_representation.copy()

    @bit_representation.setter
    def bit_representation(self, value):
        self._reset_design()
        self._bit_representation = np.array(value)
        self._redesign()

    def total_cost(self) -> float:
        n_isolated_sources = self.calculate_flows()
        if n_isolated_sources > 0:
            return self._max_possible_cost * n_isolated_sources
        else:
            return super().total_cost()

    # TODO delete method
    def change_layout(self, layout):
        self.layout = layout

    def calculate_flows(self):
        n_isolated_sources = 0
        for source, demand in self._sources.items():
            for sink, _ in self._sinks.items():
                try:
                    source = Point(x=source[0], y=source[1])
                    sink = Point(x=sink[0], y=sink[1])
                    traces = nx.shortest_path(self._design, source, sink)
                    for u, v in zip(traces, traces[1:]):
                        self._edges_props[u, v]['actual_flow'] += demand
                except nx.exception.NetworkXNoPath:
                    n_isolated_sources += 1
        return n_isolated_sources

    def nbits_required(self):
        return (len(bin(len(self.cost_model) - 1)) - 2) * len(self._edges_props) // 2

    def _update_edge(self, bits, index):
        bits = ''.join(str(bit) for bit in bits)
        pipe_props = self._bits_mapping[bits]
        if pipe_props.cost:
            edge = self._edge_by_index[index]
            self._design.add_edge(*edge)
            super()._update_edge(
                *edge,
                diameter=pipe_props.diameter,
                cost=pipe_props.cost,
                bits=bits)

    def _init_edge_by_index(self):
        indexing = self._layout.get_edge_indexing(TraversingType.BFS)
        self._edge_by_index = {index: edge for edge, index in indexing.items()}

    # TODO scale
    def _place_productive_nodes_on_layout(self):
        scale = self._layout.scale
        for pos, demand in self._sinks.items():
            position = pos[0] / scale, pos[1] / scale
            nearest = self._nearest_node(position)
            self._update_node(nearest, position=position, demand=demand)
        for pos, demand in self._sources.items():
            position = pos[0] / scale, pos[1] / scale
            nearest = self._nearest_node(position)
            self._update_node(nearest, position=position, demand=demand)

    def _redesign(self):
        substring_len = (len(bin(len(self.cost_model) - 1)) - 2)
        n = self._bit_representation.shape[0]
        for i in range(n // substring_len):
            bits = self._bit_representation[i * substring_len:(i + 1) * substring_len]
            self._update_edge(bits=bits, index=i)


class NetworkGA2d(NetworkGA):
    def _init_edge_by_index(self):
        indexing = self._layout.get_edge_indexing(TraversingType.BY_CONSTRUCTION)
        self._edge_by_index = {index: edge for edge, index in indexing.items()}

    def _redesign(self):
        substring_len = (len(bin(len(self.cost_model) - 1)) - 2)
        n, m = self._bit_representation.shape
        for i in range(n):
            for j in range(m // substring_len):
                bits = self._bit_representation[i][j * substring_len:(j + 1) * substring_len]
                self._update_edge(bits=bits, index=(i, j))


if __name__ == '__main__':
    sinks = {
        (2, 2): -120,
    }
    sources = {
        (0, 0): 10,
        (0, 1): 10,
        (0, 2): 20,
        (1, 0): 20,
        (1, 1): 10,
        (1, 2): 20,
        (2, 0): 20,
        (2, 1): 10,
    }
    #
    # network = NetworkGA(sinks, sources)
    # network.change_layout(SquareLayout(10))
    # cost_model = [PipeProps(diameter=0.0, cost=0.0), PipeProps(diameter=80.0, cost=23.0),
    #               PipeProps(diameter=100.0, cost=32.0), PipeProps(diameter=120.0, cost=50.0),
    #               PipeProps(diameter=140.0, cost=60.0), PipeProps(diameter=160.0, cost=90.0),
    #               PipeProps(diameter=180.0, cost=130.0), PipeProps(diameter=200.0, cost=170.0),
    #               PipeProps(diameter=220.0, cost=300.0), PipeProps(diameter=240.0, cost=340.0),
    #               PipeProps(diameter=260.0, cost=390.0), PipeProps(diameter=280.0, cost=430.0),
    #               PipeProps(diameter=300.0, cost=470.0), PipeProps(diameter=320.0, cost=500.0)]
    # network.cost_model = cost_model
    #
    # network.draw_pdf('net.pdf')
    from layouts import SquareLayout
    import numpy as np

    network = NetworkGA2d(sinks=sinks, sources=sources)
    layout = SquareLayout(30)
    layout.scale = 0.1
    cost_model = [PipeProps(diameter=0.0, cost=0.0),
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
                  PipeProps(diameter=320.0, cost=500.0)]
    network.layout = layout
    network.cost_model = cost_model
    # network.bit_representation = np.array(
    #     [
    #         [1,0,1,1,  1,0,1,1,  1,1,0,0,  0,1,1,0,  0,0,0,1,  0,0,1,1],
    #         [0,0,0,0,  0,1,0,1,  1,0,0,1,  0,0,0,1,  1,0,1,1,  0,0,1,0],
    #         [0,1,1,0,  1,0,1,0,  0,0,0,1,  0,1,0,0,  0,0,1,1,  0,0,1,1]
    #     ]
    # )
    network.draw_pdf('for_test.pdf')
