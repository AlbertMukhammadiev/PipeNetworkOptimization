import math
import networkx as nx
import matplotlib.pyplot as plt

from properties import *
from pprint import pprint


class Network:
    VELOCITY = 1

    def __init__(self, sinks: dict, sources: dict) -> None:
        self._sinks = sinks
        self._sources = sources
        self._cost_model = dict()
        self._edges_props = dict()
        self._nodes_props = dict()
        self._graph = nx.Graph()

    @property
    def cost_model(self):
        return self._cost_model.copy()

    @cost_model.setter
    def cost_model(self, value):
        self._cost_model = value

    def draw_in_detail(self):
        pos = {node: self._nodes_props[node].position for node in self._graph.nodes}
        nx.draw_networkx_nodes(self._graph, pos,
                               nodelist=self._nodes_props.keys(),
                               node_color='aqua',
                               node_size=200,
                               alpha=0.8)
        nx.draw_networkx_labels(self._graph, pos, self._nodes_props, font_size=6)
        nx.draw_networkx_edges(self._graph, pos)
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=self._edges_props, font_size=6)
        plt.show()

    def draw(self):
        pos = {node: self._nodes_props[node].position for node in self._graph.nodes}
        nx.draw_networkx_nodes(self._graph, pos,
                               node_color='black',
                               node_size=50)
        nx.draw_networkx_edges(self._graph, pos)
        plt.show()

    @staticmethod
    def flow_rate(diam: float, velocity: float):
        # TODO the dependence on units of measurement
        return 1 / 4 * math.pi * (diam) ** 2 * velocity / 1000 * 10


class NetworkGA(Network):
    def __init__(self, sinks, sources):
        super().__init__(sinks, sources)
        self._max_possible_cost = 0
        self._nbits_4pipe = 0
        self._layout = None
        self._edge_by_No = dict()
        self._n_isolated_nodes = 0
        self._design = ''

    @Network.cost_model.setter
    def cost_model(self, value):
        self._cost_model.clear()
        if 0 not in [props.diameter for props in value]:
            value.append(PipeProps(diameter=0, cost=0))
        sorted_props = sorted(value, key=lambda pipe: pipe.diameter)
        key_len = len(bin(len(value) - 1)) - 2
        for num in range(2 ** key_len):
            key = bin(num)[2:].zfill(key_len)
            try:
                self._cost_model[key] = sorted_props[num]
            except IndexError:
                self._cost_model[key] = sorted_props[-1]
            else:
                pass
            finally:
                pass

        self._max_possible_cost = self._max_cost()
        self._nbits_4pipe = (len(bin(len(self._cost_model) - 1)) - 2)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value):
        numeration = value.numerated_edges
        for edge in value.edges:
            u, v = edge
            self._graph.add_edge(u, v)
            self._nodes_props[u] = NodeProps(demand=0, position=u)
            self._nodes_props[v] = NodeProps(demand=0, position=v)
            self._edges_props[(u, v)] = EdgeProps(length=1, No=numeration[edge])
            self._edge_by_No[numeration[edge]] = edge

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, value):
        self._reset()
        bits = ''.join(str(bit) for bit in value)
        substrings = [bits[i:i + self._nbits_4pipe] for i in range(0, len(bits), self._nbits_4pipe)]
        for i, substring in enumerate(substrings):
            pipe_props = self._cost_model[substring]
            if pipe_props.cost:
                edge = self._edge_by_No[i]
                self._edges_props[edge].diameter = pipe_props.diameter
                self._edges_props[edge].cost = pipe_props.cost
                self._edges_props[edge].actual_flow = 0
                self._edges_props[edge].flow_rate = Network.flow_rate(pipe_props.diameter, self.VELOCITY)
                self._graph.add_edge(*edge)

    def build_flows(self):
        for source in self._sources:
            for sink, sink_props in self._sinks.items():
                try:
                    path = nx.shortest_path(self._graph, sink, source)
                    for u, v in zip(path, path[1:]):
                        self._edges_props[(u, v)].actual_flow += sink_props.demand
                        # self._current_state[(u, v)]['actual_flow'] += sink_props.demand
                except nx.exception.NetworkXNoPath:
                    self._n_isolated_nodes += 1

    def penalty_cost(self):
        penalty = 0
        for edge in self._graph.edges():
            # pprint(self._graph.edges)
            # pprint(self._edges_props)
            try:
                props = self._edges_props[edge]
            except KeyError:
                u, v = edge
                props = self._edges_props[(v, u)]
            rate = props.actual_flow / props.flow_rate
            if rate > 1:
                penalty += props.cost * math.ceil(rate) * props.length * 3
        # for props in self._current_state.values():
        #     rate = props['actual_flow'] / props['flow_rate']
        #     if rate > 1:
        #         penalty += props['cost'] * math.ceil(rate) * props['length'] * 3
        return penalty

    def total_cost(self) -> float:
        if self._n_isolated_nodes > 0:
            return self._max_possible_cost * self._n_isolated_nodes

        total_cost = 0
        penalty = self.penalty_cost()
        for edge in self._graph.edges():
            try:
                props = self._edges_props[edge]
            except KeyError:
                u, v = edge
                props = self._edges_props[(v, u)]
            total_cost += props.cost * props.length

        return total_cost + penalty

    def _max_cost(self):
        most_expensive = max([pipe.cost for pipe in self._cost_model.values()])
        weights = [props.length * most_expensive for props in self._edges_props.values()]
        return sum(weights)

    def nbits_required(self):
        return len(self._edges_props) * self._nbits_4pipe

    def _reset(self):
        # self._edges_props = dict()
        edges = list(self._graph.edges())
        self._graph.remove_edges_from(edges)


if __name__ == '__main__':
    from layouts import SquareLayout, HexagonLayout
    from data_context import DataContext
    from pprint import pprint

    cost_model = [PipeProps(diameter=0.0, cost=0.0), PipeProps(diameter=80.0, cost=23.0),
                  PipeProps(diameter=100.0, cost=32.0), PipeProps(diameter=120.0, cost=50.0),
                  PipeProps(diameter=140.0, cost=60.0), PipeProps(diameter=160.0, cost=90.0),
                  PipeProps(diameter=180.0, cost=130.0), PipeProps(diameter=200.0, cost=170.0),
                  PipeProps(diameter=220.0, cost=300.0), PipeProps(diameter=240.0, cost=340.0),
                  PipeProps(diameter=260.0, cost=390.0), PipeProps(diameter=280.0, cost=430.0),
                  PipeProps(diameter=300.0, cost=470.0), PipeProps(diameter=320.0, cost=500.0)]
    net = NetworkGA(dict(), dict())
    net.layout = SquareLayout(2, 3)
    net.cost_model = cost_model


    path = 'projects/square_layout/'
    data_context = DataContext(path)
    # network = Network(layout)
    individual = [
        0, 1, 0, 0,  # 1
        0, 0, 0, 0,  # 2
        0, 0, 1, 1,  # 3
        0, 0, 1, 0,  # 4
        0, 0, 0, 1,  # 5
        0, 0, 0, 0,  # 6
        0, 0, 0, 0,  # 7
        0, 1, 0, 1,  # 8
        0, 0, 0, 0,  # 9
        0, 0, 0, 0,  # 10
        0, 0, 1, 1,  # 11
        0, 0, 1, 0]  # 12

    net.redesign(individual)
    net.draw_in_detail()
    # print(net.total_cost(individual))
    # print(net._max_cost())



