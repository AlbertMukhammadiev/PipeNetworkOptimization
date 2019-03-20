import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])


class Network:
    def __init__(self, layout, cost_model):
        self._edges = layout.edges
        self._nodes = layout.nodes
        self._design = nx.Graph()
        self._design.add_nodes_from(self._nodes)
        self._sinks = set()
        self._sources = set()
        self._init_cost_model(cost_model)
        self._init_max_possible_cost()

    def _init_cost_model(self, cost_model):
        self._cost_model = list()
        if 0 not in [props['diam'] for props in cost_model]:
            self._cost_model.append(dict(diameter=0, cost=0))
        self._cost_model.extend(cost_model)

    def _init_max_possible_cost(self):
        self._most_expensive = max([pipe['cost'] for pipe in self._cost_model])
        lengths = [props['len'] for props in self._edges.values()]
        self._max_possible_cost = sum(self._most_expensive * np.array(lengths)) / 2

    @staticmethod
    def flow_rate(diameter: float, velocity: float):
        return 1 / 4 * math.pi * diameter ** 2 * velocity / 1000

    def add_sink(self, pos, demand):
        if demand < 0:
            node = self._nearest_to(point=pos)
            self._update_node(node, pos=pos, demand=demand)
            self._sinks.add(node)
            print(f'--- the sink was set at point {pos}')

    def add_source(self, pos, demand):
        if demand > 0:
            node = self._nearest_to(point=pos)
            self._update_node(node, pos=pos, demand=demand)
            self._sources.add(node)
            print(f'--- the source was set at point {pos}')

    def development_cost(self):
        return self.preliminary_cost() + self.penalty_cost()

    def preliminary_cost(self):
        costs = [self._edges[edge]['cost'] for edge in self._design.edges]
        lengths = [self._edges[edge]['len'] for edge in self._design.edges]
        return sum(np.array(costs) * np.array(lengths))

    def penalty_cost(self):
        lengths = [self._edges[edge]['len'] for edge in self._design.edges]
        redevelopment_cost = []
        for edge in self._design.edges:
            props = self._edges[edge]
            rate = math.floor(props['flow_rate'] / props['max_flow_rate'])
            redevelopment_cost.append(rate * self._most_expensive)
        return sum(np.array(redevelopment_cost) * np.array(lengths))

    def distance(self, u, v):
        if not isinstance(u, Point):
            u = self._nodes[u]['pos']
        if not isinstance(v, Point):
            v = self._nodes[v]['pos']
        return math.hypot(v.y - u.y, v.x - u.x)

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

    def _update_node(self, node, **kwargs):
        for key, value in kwargs.items():
            self._nodes[node][key] = value

    def _update_edge(self, u, v, **kwargs):
        for key, value in kwargs.items():
            self._edges[u, v][key] = value

    def _nearest_to(self, point):
        return min(
            self._nodes,
            key=lambda node: self.distance(point, node))

    def _drawing_configurations(self):
        pos = {node: props['pos'] for node, props in self._nodes.items()}
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

        for node, props in self._nodes.items():
            label = f'pos: ({props["pos"].x}, {props["pos"].y})\n' \
                f'demand: {props["demand"]}'
            kwargs['labels'][node] = label
            if self._nodes[node]['demand'] > 0:
                kwargs['node_color'].append('lightgreen')
            elif self._nodes[node]['demand'] < 0:
                kwargs['node_color'].append('salmon')
            else:
                kwargs['node_color'].append('lightgoldenrodyellow')

        for edge in self._design.edges:
            props = self._edges[edge]
            label = f'l/d/c: {props["len"]}/ {props["diam"]}/ {props["cost"]}\n' \
                f'flow: {props["flow_rate"]}/{round(props["max_flow_rate"], 2)}'
            kwargs['edge_labels'][edge] = label

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
            self._edge_by_index[props['cindex']] = edge

    def _init_ga_attributes(self):
        self._chromosome = None
        self._mapped_cost_model = dict()
        self._b_len = (len(bin(len(self._cost_model) - 1)) - 2)
        sorted_props = sorted(self._cost_model, key=lambda pipe: pipe['diam'])
        for number in range(2 ** self._b_len):
            key = bin(number)[2:].zfill(self._b_len)
            try:
                self._mapped_cost_model[key] = sorted_props[number]
            except IndexError:
                self._mapped_cost_model[key] = sorted_props[-1]

    @property
    def chromosome(self):
        return self._chromosome.copy()

    @chromosome.setter
    def chromosome(self, value):
        self._reset_design()
        self._chromosome = value
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
                        self._edges[u, v]['flow_rate'] += self._nodes[source]['demand']
                except nx.exception.NetworkXNoPath:
                    self._n_isolated_sources += 1

    def _redesign(self):
        for i in range(self.chromosome_shape // self._b_len):
            bits = self._chromosome[i * self._b_len:(i + 1) * self._b_len]
            self._redesign_edge(index=i, gene=bits)
        self._calculate_flows()

    def _redesign_edge(self, index, gene):
        u, v = self._edge_by_index[index]
        props = self._mapped_cost_model[''.join(map(str, gene))]
        if props['diam']:
            self._design.add_edge(u, v)
            super()._update_edge(
                u, v,
                diam=props['diam'],
                cost=props['cost'],
                max_flow_rate=Network.flow_rate(props['diam'], 1),
                gene=gene
            )

    def _reset_design(self):
        edges = list(self._design.edges)
        self._design.remove_edges_from(edges)
        for edge in self._edges:
            self._edges[edge]['flow_rate'] = 0

    def _drawing_configurations(self):
        pos, kwargs = super()._drawing_configurations()

        for edge in self._design.edges:
            props = self._edges[edge]
            label = f'\ngene: {props["gene"]}\n' \
                f'cindex: {props["cindex"]}\n' \
                f'indexBFS: {props["indexBFS"]}'
            kwargs['edge_labels'][edge] += label
        return pos, kwargs


class NetworkGA2D(NetworkGA):
    def _init_edge_by_index(self):
        self._edge_by_index = dict()
        for edge, props in self._edges.items():
            self._edge_by_index[props['cindex2d']] = edge

        n = max(i for i, j in self._edge_by_index.keys()) + 1
        m = max(j for i, j in self._edge_by_index.keys()) + 1
        self._chromosome_shape = (n, m * self._b_len)

    def _redesign(self):
        n, m = self.chromosome_shape
        for i in range(n):
            for j in range(m // self._b_len):
                bits = self._chromosome[i][j * self._b_len:(j + 1) * self._b_len]
                self._redesign_edge(index=(i, j), gene=bits)
        self._calculate_flows()

    def _drawing_configurations(self):
        pos, kwargs = super()._drawing_configurations()
        for edge in self._design.edges:
            props = self._edges[edge]
            label = f'\ncindex2d: {props["cindex2d"]}'
            kwargs['edge_labels'][edge] += label
        return pos, kwargs
