from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from numpy import arange
from enum import Enum, auto

Edge = namedtuple('Edge', ['u', 'v'])
Point = namedtuple('Point', ['x', 'y'])
Index = namedtuple('Index', ['i', 'j'])

class TraversingType(Enum):
    BFS = auto()
    BY_CONSTRUCTION = auto()


class Layout:
    def __init__(self, n):
        self._indexing_by_construction = dict()
        self._current_edge_index = Index(0, 0)

        self._edges = set()
        self._n = n
        self._scale = 1
        self._drawing_kwargs = {
            'node_color': 'aqua',
            'node_size': 50,
            'alpha': 0.8,
            'font_size': 1,
            'rotate': False,
            'width': 0.1,
        }

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value > 0:
            self._scale = valuegigi

    @property
    def edges(self):
        return self._edges.copy()

    def get_edge_indexing(self, traversing_type):
        if traversing_type is TraversingType.BFS:
            graph = nx.Graph()
            graph.add_edges_from(self._edges)
            start_node = sorted(graph.nodes, reverse=True).pop()
            path_bfs = nx.edge_bfs(graph, start_node)
            indexing = {}
            for i, edge in enumerate(path_bfs):
                u, v = edge
                indexing[u, v] = i
                indexing[v, u] = i
            return indexing
        elif traversing_type is TraversingType.BY_CONSTRUCTION:
            return self._indexing_by_construction.copy()

    def add_edge(self, u, v):
        self._indexing_by_construction[u, v] = self._current_edge_index
        self._indexing_by_construction[v, u] = self._current_edge_index
        self._current_edge_index = Index(
            self._current_edge_index.i,
            self._current_edge_index.j + 1)
        self._edges.add(Edge(u, v))

    def next_by_column(self, point):
        if point.x + 1 == self._n:
            raise IndexError
        return Point(
            x=point.x + 1,
            y=point.y)

    def next_by_row(self, point):
        if point.y + 1 == self._n:
            raise IndexError
        return Point(
            y=point.y + 1,
            x=point.x)

    def draw_for_test(self):
        import random
        drawing_kwargs = {
            'node_color': 'aqua',
            'node_size': 300,
            'alpha': 1.,
            'font_size': 6,
            'rotate': False,
            'width': 0.5,
        }
        G = nx.Graph()
        for edge in self._edges:
            u, v = edge
            G.add_edge(u, v)

        pos = {node: (node.x, node.y) for node in G.nodes}
        indexing = self.get_edge_indexing(TraversingType.BFS)
        edge_labels = {}
        for edge in G.edges:
            index_2d = self._indexing_by_construction[edge]
            index_bfs = indexing[edge]
            key = bin(random.randint(0, 15))[2:].zfill(4)
            edge_labels[edge] = f'2d: {index_2d.i}, {index_2d.j}\nbfs: {index_bfs}\npipe: {key}'

        labels = {node: f'{node.x}, {node.y}' for node in G.nodes}

        plt.axis('equal')
        nx.draw_networkx_nodes(G, pos, **drawing_kwargs)
        nx.draw_networkx_labels(G, pos, labels=labels, **drawing_kwargs)
        nx.draw_networkx_edges(G, pos, **drawing_kwargs)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, **drawing_kwargs)
        plt.savefig('test_layout.pdf', format='pdf')
        plt.close()

    def draw(self):
        G = nx.Graph()
        for edge in self._edges:
            u, v = edge
            G.add_edge(u, v)

        pos = {node: (node.x, node.y) for node in G.nodes}
        edge_labels = {edge: self._indexing_by_construction[edge] for edge in G.edges}

        plt.axis('equal')
        nx.draw_networkx_nodes(G, pos, **self._drawing_kwargs)
        nx.draw_networkx_labels(G, pos, **self._drawing_kwargs)
        nx.draw_networkx_edges(G, pos, **self._drawing_kwargs)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, **self._drawing_kwargs)
        plt.savefig('test_layout.pdf', format='pdf')
        plt.close()


class SquareLayout(Layout):
    def __init__(self, n):
        super().__init__(n=n)
        self.create_connections()

    def create_connections(self):
        for i in range(self._n - 1):
            self._current_edge_index = Index(i, 0)
            self.construct_row(i)

    def construct_row(self, y_k):
        for x_k in range(self._n - 1):
            self.construct_connections_from(Point(x=x_k, y=y_k))

    def construct_connections_from(self, point):
        v = self.next_by_row(point)
        self.add_edge(point, v)
        u = self.next_by_column(point)
        self.add_edge(point, u)


class HexagonLayout(Layout):
    def __init__(self, n):
        super().__init__(n=n)
        self.create_connections()

    def create_connections(self):
        for i in range(self._n - 1):
            self._current_edge_index = Index(i, 0)
            self.construct_row(i)

    def construct_row(self, y_k):
        for x_k in range(self._n - 1):
            self.construct_connections_from(Point(x=x_k, y=y_k))

    def construct_connections_from(self, point):
        v = self.next_by_row(point)
        self.add_edge(point, v)
        u = self.next_by_column(point)
        self.add_edge(point, u)


if __name__ == '__main__':
    # # layout = SquareLayout(10)
    # # layout.draw()
    # import random
    # from pprint import pprint
    #
    # def individual1(n, m):
    #     return [random.choices([0, 1], k=m) for _ in range(n)]
    #
    # def individual2(n, m):
    #     return [random.choices([2, 3], k=m) for _ in range(n)]
    #
    # def cx_two_point_2d(ind1, ind2):
    #     n = min(len(ind1), len(ind2))
    #     m = min(len(ind1[0]), len(ind2[0]))
    #     cx_points = random.choices(range(m), k=2)
    #     print(cx_points)
    #     cy_points = random.choices(range(n), k=2)
    #     print(cy_points)
    #     cx_point1, cx_point2 = min(cx_points), max(cx_points) + 1
    #     cy_point1, cy_point2 = min(cy_points), max(cy_points) + 1
    #     for i in range(cy_point1, cy_point2):
    #         ind1[i][cx_point1:cx_point2], ind2[i][cx_point1:cx_point2] \
    #             = ind2[i][cx_point1:cx_point2], ind1[i][cx_point1:cx_point2]
    #
    #     return ind1, ind2
    #
    #
    # a = individual1(5, 10)
    # b = individual2(5, 10)
    # pprint(a)
    # print()
    # pprint(b)
    # print()
    #
    # cx_two_point_2d(a, b)
    # print()
    # pprint(a)
    # print()
    # pprint(b)

    layout = SquareLayout(4)
    layout.draw_for_test()


#
# class Layout1:
#     def __init__(self):
#         self._layout = nx.Graph()
#         self._edges = set()
#
#     @property
#     def numerated_edges(self):
#         graph = nx.Graph()
#         graph.add_edges_from(self._edges)
#         path_bfs = nx.edge_bfs(graph, (0, 0))
#         numerated_edges = {}
#         for i, edge in enumerate(path_bfs):
#             u, v = edge
#             numerated_edges[u, v] = i
#             numerated_edges[v, u] = i
#         return numerated_edges
#
#     @property
#     def edges(self):
#         return self._edges.copy()
#
#     @edges.setter
#     def edges(self, ls):
#         if not isinstance(self, (SquareLayout, HexagonLayout)):
#             self._edges = ls
#
#     def draw(self):
#         pos = {node: node for node in self._layout.nodes}
#         nx.draw_networkx_nodes(self._layout, pos,
#                                node_color='black',
#                                node_size=50)
#         nx.draw_networkx_edges(self._layout, pos)
#         plt.show()
#
#     def draw_with_node_labels(self):
#         G = nx.Graph()
#         for edge in self._edges:
#             u, v = edge
#             G.add_edge(u, v)
#
#         pos = {node: node for node in G.nodes}
#         labels = {node: f'{node[0]}, {node[1]}' for node in G.nodes}
#         nx.draw_networkx_nodes(G, pos,
#                                nodelist=labels.keys(),
#                                node_color='aqua',
#                                node_size=400)
#         nx.draw_networkx_labels(G, pos, labels, font_size=10)
#         nx.draw_networkx_edges(G, pos)
#         plt.show()
#
#     def draw_with_node_labels2(self):
#         pos = {node: node for node in self._layout.nodes}
#         labels = {node: f'{node[0]}, {node[1]}' for node in self._layout.nodes}
#         nx.draw_networkx_nodes(self._layout, pos,
#                                nodelist=labels.keys(),
#                                node_color='aqua',
#                                node_size=400)
#         nx.draw_networkx_labels(self._layout, pos, labels, font_size=10)
#         nx.draw_networkx_edges(self._layout, pos)
#         plt.show()
#
#     def draw_with_edge_labels(self):
#         pos = {node: node for node in self._layout.nodes}
#         labels = {node: node for node in self._layout.nodes}
#         nx.draw_networkx_nodes(self._layout, pos,
#                                nodelist=labels.keys(),
#                                node_color='aqua',
#                                node_size=200,
#                                alpha=0.8)
#         nx.draw_networkx_labels(self._layout, pos, labels, font_size=10)
#         edge_labels = self.numerated_edges
#         nx.draw_networkx_edge_labels(self._layout, pos, edge_labels=edge_labels)
#         nx.draw_networkx_edges(self._layout, pos)
#         plt.show()
#
#     def right(self, x, y):
#         u, v = (x, y), (x + 1, y)
#         edge = Edge(u, v)
#         self._layout.add_edge(u, v)
#         self._edges.add(edge)
#
#
# class HexagonLayout(Layout):
#     def __init__(self, n_xs, n_ys):
#         super().__init__()
#         for x in range(n_xs - 1):
#             self.right(x, 0)
#
#         for y in range(1, n_ys):
#             self.build_lvl(y, n_xs - 1)
#
#     def down_right(self, i, j):
#         u, v = (i, j), (i + 1, j - 1)
#         edge = Edge(u, v)
#         self._layout.add_edge(u, v)
#         self._edges.add(edge)
#
#     def down_left(self, i, j):
#         u, v = (i, j), (i - 1, j - 1)
#         edge = Edge(u, v)
#         self._layout.add_edge(u, v)
#         self._edges.add(edge)
#
#     def build_lvl(self, y, n_xs):
#         k = 0
#         while True:
#             if k == n_xs:
#                 break
#
#             self.down_right(k, y)
#             self.right(k, y)
#             k += 1
#
#             if k == n_xs:
#                 break
#
#             self.right(k, y)
#             k += 1
#
#             if k == n_xs:
#                 break
#
#             self.right(k, y)
#             k += 1
#
#             if k == n_xs:
#                 break
#
#             self.down_left(k, y)
#             self.right(k, y)
#             k += 1
#
#
# class SquareLayout(Layout):
#     def __init__(self, n_xs, n_ys):
#         super().__init__()
#         for x in range(n_xs - 1):
#             self.right(x, 0)
#
#         for y in range(1, n_ys):
#             self.build_lvl(y, n_xs - 1)
#
#     def down(self, x, y):
#         u, v = (x, y), (x, y - 1)
#         edge = Edge(u, v)
#         self._layout.add_edge(u, v)
#         self._edges.add(edge)
#
#     def build_lvl(self, y, n_xs):
#         for x in range(n_xs):
#             self.right(x, y)
#             self.down(x, y)
#         self.down(n_xs, y)
#
#
# def SquareLayout1:
#
#
#
# if __name__ == '__main__':
#     layout = HexagonLayout(3, 3)
#
#     from pprint import pprint
#     # pprint(layout.numerated_edges)
#     layout.draw_with_edge_labels()