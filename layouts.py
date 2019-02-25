from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from numpy import arange

Edge = namedtuple('Edge', ['u', 'v'])

class Layout:
    def __init__(self):
        self._layout = nx.Graph()
        self._edges = set()

    @property
    def numerated_edges(self):
        return dict(zip(self._edges, range(len(self._edges))))

    @property
    def edges(self):
        return self._edges.copy()

    @edges.setter
    def edges(self, ls):
        if not isinstance(self, (SquareLayout, HexagonLayout)):
            self._edges = ls

    def draw(self):
        pos = {node: node for node in self._layout.nodes}
        nx.draw_networkx_nodes(self._layout, pos,
                               node_color='black',
                               node_size=50)
        nx.draw_networkx_edges(self._layout, pos)
        plt.show()

    def draw_with_node_labels(self):
        G = nx.Graph()
        for edge in self._edges:
            u, v = edge
            G.add_edge(u, v)

        pos = {node: node for node in G.nodes}
        labels = {node: f'{node[0]}, {node[1]}' for node in G.nodes}
        nx.draw_networkx_nodes(G, pos,
                               nodelist=labels.keys(),
                               node_color='aqua',
                               node_size=400)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        nx.draw_networkx_edges(G, pos)
        plt.show()

    def draw_with_node_labels2(self):
        pos = {node: node for node in self._layout.nodes}
        labels = {node: f'{node[0]}, {node[1]}' for node in self._layout.nodes}
        nx.draw_networkx_nodes(self._layout, pos,
                               nodelist=labels.keys(),
                               node_color='aqua',
                               node_size=400)
        nx.draw_networkx_labels(self._layout, pos, labels, font_size=10)
        nx.draw_networkx_edges(self._layout, pos)
        plt.show()

    def draw_with_edge_labels(self):
        pos = {node: node for node in self._layout.nodes}
        labels = {node: node for node in self._layout.nodes}
        nx.draw_networkx_nodes(self._layout, pos,
                               nodelist=labels.keys(),
                               node_color='aqua',
                               node_size=200,
                               alpha=0.8)
        nx.draw_networkx_labels(self._layout, pos, labels, font_size=10)
        edge_labels = self.numerated_edges
        nx.draw_networkx_edge_labels(self._layout, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(self._layout, pos)
        plt.show()

    def right(self, x, y):
        u, v = (x, y), (x + 1, y)
        edge = Edge(u, v)
        self._layout.add_edge(u, v)
        self._edges.add(edge)


class HexagonLayout(Layout):
    def __init__(self, n_xs, n_ys):
        super().__init__()
        for x in range(n_xs - 1):
            self.right(x, 0)

        for y in range(1, n_ys):
            self.build_lvl(y, n_xs - 1)

    def down_right(self, i, j):
        u, v = (i, j), (i + 1, j - 1)
        edge = Edge(u, v)
        self._layout.add_edge(u, v)
        self._edges.add(edge)

    def down_left(self, i, j):
        u, v = (i, j), (i - 1, j - 1)
        edge = Edge(u, v)
        self._layout.add_edge(u, v)
        self._edges.add(edge)

    def build_lvl(self, y, n_xs):
        k = 0
        while True:
            if k == n_xs:
                break

            self.down_right(k, y)
            self.right(k, y)
            k += 1

            if k == n_xs:
                break

            self.right(k, y)
            k += 1

            if k == n_xs:
                break

            self.right(k, y)
            k += 1

            if k == n_xs:
                break

            self.down_left(k, y)
            self.right(k, y)
            k += 1


class SquareLayout(Layout):
    def __init__(self, n_xs, n_ys):
        super().__init__()
        for x in range(n_xs - 1):
            self.right(x, 0)

        for y in range(1, n_ys):
            self.build_lvl(y, n_xs - 1)

    def down(self, x, y):
        u, v = (x, y), (x, y - 1)
        edge = Edge(u, v)
        self._layout.add_edge(u, v)
        self._edges.add(edge)

    def build_lvl(self, y, n_xs):
        for x in range(n_xs):
            self.right(x, y)
            self.down(x, y)
        self.down(n_xs, y)


if __name__ == '__main__':
    layout = HexagonLayout(3, 3)

    from pprint import pprint
    # pprint(layout.numerated_edges)
    layout.draw_with_edge_labels()