import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import namedtuple
PipeProps = namedtuple('PipeProps', ['diameter', 'cost'])
Point = namedtuple('Point', ['x', 'y'])


class Network:
    def __init__(self, layout, cost_model):
        self._design = nx.Graph()
        self._edges = layout.edges
        self._nodes = layout.nodes
        self._sinks = set()
        self._sources = set()
        self._init_cost_model(cost_model)
        self._init_max_possible_cost()

    def _init_cost_model(self, cost_model):
        self._cost_model = list()
        if 0 not in [props['diameter'] for props in cost_model]:
            self._cost_model.append(dict(diameter=0, cost=0))
        self._cost_model.extend(cost_model)

    def _init_max_possible_cost(self):
        self._most_expensive = max([pipe['cost'] for pipe in self._cost_model])
        lengths = [props['length'] for props in self._edges.values()]
        self._max_possible_cost = sum(self._most_expensive * np.array(lengths)) / 2

    @staticmethod
    def flow_rate(diameter: float, velocity: float):
        return 1 / 4 * math.pi * diameter ** 2 * velocity / 1000

    def add_sink(self, position, demand):
        if demand < 0:
            node = self._nearest_to(point=position)
            self._update_node(node, position=position, demand=demand)
            self._sinks.add(node)
            print(f'--- the sink was set at point {position}')

    def add_source(self, position, demand):
        if demand > 0:
            node = self._nearest_to(point=position)
            self._update_node(node, position=position, demand=demand)
            self._sources.add(node)
            print(f'--- the source was set at point {position}')

    def development_cost(self):
        return self.preliminary_cost() + self.penalty_cost()

    def preliminary_cost(self):
        costs = [self._edges[edge]['cost'] for edge in self._design.edges]
        lengths = [self._edges[edge]['length'] for edge in self._design.edges]
        return sum(np.array(costs) * np.array(lengths))

    def penalty_cost(self):
        lengths = [self._edges[edge]['length'] for edge in self._design.edges]
        redevelopment_cost = []
        for edge in self._design.edges:
            props = self._edges[edge]
            rate = math.floor(props['actual_flow'] / props['flow_rate'])
            redevelopment_cost.append(rate * self._most_expensive)
        return sum(np.array(redevelopment_cost) * np.array(lengths))

    def distance(self, u, v):
        if not isinstance(u, Point):
            u = self._nodes[u]['position']
        if not isinstance(v, Point):
            v = self._nodes[v]['position']
        return math.hypot(v.y - u.y, v.x - u.x)

    def draw(self):
        pos, kwargs = self._drawing_configurations()
        nx.draw_networkx_nodes(self._design, pos, **kwargs)
        nx.draw_networkx_edges(self._design, pos, **kwargs)
        plt.show()

    def draw_in_detail(self):
        self._design.add_edges_from(self._edges)
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

    def _update_node(self, node, **kwargs):
        for key, value in kwargs.items():
            self._nodes[node][key] = value

    def _update_edge(self, u, v, **kwargs):
        for key, value in kwargs.items():
            self._edges[u, v][key] = value

    def _nearest_to(self, point):
        return min(
            self._nodes,
            key=lambda node: self.distance(point, node)
        )

    def _drawing_configurations(self):
        pos = {node: self._nodes[node]['position'] for node in self._design.nodes}
        kwargs = {
            'node_color': [],
            'node_size': 50,
            'node_shape': 's',
            'alpha': 1,
            'font_size': 1,
            'rotate': False,
            'width': 0.2,
            'labels': {},
            'edge_labels': {}}

        for node in self._design.nodes:
            if self._nodes[node]['demand'] > 0:
                kwargs['node_color'].append('lightgreen')
            elif self._nodes[node]['demand'] < 0:
                kwargs['node_color'].append('salmon')
            else:
                kwargs['node_color'].append('lightgoldenrodyellow')
        for edge in self._design.edges:
            props = self._edges[edge]
            label = f'l:{props["length"]} d:{props["diameter"]} c:{props["cost"]}\n' \
                f'flow: {props["actual_flow"]}/{round(props["flow_rate"], 2)}\n' \
                f'index: {props["constr_No"]}/{props["2d"]}/{props["bfs_ind"]}\n' \
                f'gene: {props["gene"]}'
            kwargs['edge_labels'][edge] = label
        for node in self._design.nodes:
            props = self._nodes[node]
            label = f'pos: ({props["position"].x}, {props["position"].y})\n' \
                f'demand: {props["demand"]}'
            kwargs['labels'][node] = label

        return pos, kwargs


class NetworkGA(Network):
    def __init__(self, layout, cost_model):
        super().__init__(layout=layout, cost_model=cost_model)
        self._init_ga_attributes()
        self._init_edge_by_index()

    def _init_edge_by_index(self):
        self._chromosome_shape = len(self._edges) * self._b_len // 2
        self._edge_by_index = dict()
        for edge, props in self._edges.items():
            self._edge_by_index[props['constr_No']] = edge

    def _init_ga_attributes(self):
        self._c = None
        self._mapped_cost_model = dict()
        self._b_len = (len(bin(len(self._cost_model) - 1)) - 2)
        sorted_props = sorted(self._cost_model, key=lambda pipe: pipe['diameter'])
        for number in range(2 ** self._b_len):
            key = bin(number)[2:].zfill(self._b_len)
            try:
                self._mapped_cost_model[key] = sorted_props[number]
            except IndexError:
                self._mapped_cost_model[key] = sorted_props[-1]

    @property
    def chromosome(self):
        return self._c.copy()

    @chromosome.setter
    def chromosome(self, value):
        self._reset_design()
        self._c = value
        self._redesign()

    @property
    def chromosome_shape(self):
        return self._chromosome_shape

    def development_cost(self) -> float:
        if self._n_isolated_sources > 0:
            return self._max_possible_cost * self._n_isolated_sources
        else:
            return super().development_cost()

    def _calculate_flows(self):
        self._n_isolated_sources = 0
        for source in self._sources:
            for sink in self._sinks:
                try:
                    traces = nx.shortest_path(self._design, source, sink)
                    for u, v in zip(traces, traces[1:]):
                        self._edges[u, v]['actual_flow'] += self._nodes[source]['demand']
                except nx.exception.NetworkXNoPath:
                    self._n_isolated_sources += 1

    def _redesign(self):
        for i in range(self.chromosome_shape // self._b_len):
            bits = self._c[i * self._b_len:(i + 1) * self._b_len]
            self._redesign_edge(index=i, gene=bits)
        self._calculate_flows()

    def _redesign_edge(self, index, gene):
        u, v = self._edge_by_index[index]
        props = self._mapped_cost_model[''.join(map(str, gene))]
        if props['diameter']:
            self._design.add_edge(u, v)
            super()._update_edge(
                u, v,
                diameter=props['diameter'],
                cost=props['cost'],
                flow_rate=Network.flow_rate(props['diameter'], 1),
                gene=gene
            )

    def _reset_design(self):
        edges = list(self._design.edges)
        self._design.remove_edges_from(edges)
        for edge in self._edges:
            self._edges[edge]['actual_flow'] = 0


class NetworkGA2D(NetworkGA):
    def _init_edge_by_index(self):
        self._edge_by_index = dict()
        for edge, props in self._edges.items():
            self._edge_by_index[props['constr_No']] = edge

        n = max(i for i, j in self._edge_by_index.keys()) + 1
        m = max(j for i, j in self._edge_by_index.keys()) + 1
        self._chromosome_shape = (n, m * self._b_len)

    def _redesign(self):
        n, m = self.chromosome_shape
        for i in range(n):
            for j in range(m // self._b_len):
                bits = self._c[i][j * self._b_len:(j + 1) * self._b_len]
                self._redesign_edge(index=(i, j), gene=bits)
