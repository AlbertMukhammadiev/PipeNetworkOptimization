import math
import networkx as nx
import matplotlib.pyplot as plt
from network import Point


class Layout:
    def __init__(self):
        self._edges = dict()
        self._nodes = dict()
        self._init_indexing_tools()
        self._init_scale()

    def _init_indexing_tools(self):
        self._current_index = 0

    def _init_scale(self):
        self._scale = 1
        self._round_dec = 3

    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def edges(self):
        return self._edges.copy()

    def distance(self, u, v):
        if not isinstance(u, Point):
            u = self._nodes[u]['position']
        if not isinstance(v, Point):
            v = self._nodes[v]['position']
        return math.hypot(v.y - u.y, v.x - u.x)

    def draw(self):
        pos, _ = self._drawing_configurations()
        graph = nx.Graph()
        graph.add_edges_from(self._edges)
        plt.axis('equal')
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        plt.show()
        plt.close()

    def draw_in_detail(self):
        pos, kwargs = self._drawing_configurations()
        graph = nx.Graph()
        graph.add_edges_from(self._edges)
        plt.axis('equal')
        nx.draw_networkx_nodes(graph, pos, **kwargs)
        nx.draw_networkx_labels(graph, pos, **kwargs)
        nx.draw_networkx_edges(graph, pos, **kwargs)
        nx.draw_networkx_edge_labels(graph, pos, **kwargs)
        plt.show()
        plt.close()

    def draw_pdf(self, path):
        pos, kwargs = self._drawing_configurations()
        graph = nx.Graph()
        graph.add_edges_from(self._edges)
        plt.axis('equal')
        nx.draw_networkx_nodes(graph, pos, **kwargs)
        nx.draw_networkx_labels(graph, pos, **kwargs)
        nx.draw_networkx_edges(graph, pos, **kwargs)
        nx.draw_networkx_edge_labels(graph, pos, **kwargs)
        plt.savefig(path, format='pdf')
        plt.close()

    def add_node(self, node, position):
        if node not in self._nodes:
            self._nodes[node] = dict(position=position, demand=0)
            print(f'--- node {node} with position {position} was added')
        else:
            self._nodes[node]['position'] = position
            print(f'--- position of node {node}'
                  f'was changed to {position}')

    def add_edge(self, u, v):
        self.add_node(u, u)
        self.add_node(v, v)
        if (u, v) not in self._edges and (v, u) not in self._edges:
            props = dict(
                constr_No=self._current_index,
                length=self._scale,
            )
            self._edges[u, v] = props
            self._edges[v, u] = props
            print(f'--- edge {u}-{v} {v}-{u} was added'
                  f' with index ({self._current_index})')
            self._update_current_index()

    def _index_edges(self):
        self._index_edges_bfs()

    def _index_edges_bfs(self):
        graph = nx.Graph()
        graph.add_edges_from(self._edges)
        start_node = sorted(graph.nodes, reverse=True).pop()
        path_bfs = nx.edge_bfs(graph, start_node)
        for i, edge in enumerate(path_bfs):
            self._edges[edge]['bfs_ind'] = i

    def _update_current_index(self):
        self._current_index += 1

    def _drawing_configurations(self):
        pos = {node: props['position'] for node, props in self._nodes.items()}
        kwargs = {
            'node_color': 'lightgoldenrodyellow',
            'node_size': 50,
            'node_shape': 's',
            'alpha': 1,
            'font_size': 1,
            'width': 0.2,
            'rotate': False,
            'labels': {},
            'edge_labels': {}}

        for edge, props in self._edges.items():
            label = f'l: {props["length"]}\n' \
                f'index: {props["constr_No"]}/ {props["2d"]}'
            kwargs['edge_labels'][edge] = label
        for node, props in self._nodes.items():
            label = f'pos: ({props["position"].x}, {props["position"].y})\n'
            kwargs['labels'][node] = label

        return pos, kwargs


class RegularGrid(Layout):
    def __init__(self, n_rows, n_columns, scale=1):
        super().__init__()
        self._n = n_rows
        self._m = n_columns
        self._scale = scale
        self._start = Point(0, 0)
        self._nodes[self._start] = {'position': self._start}
        self._init_vectors()
        self._create()
        self._index_edges()

    def _init_indexing_tools(self):
        self._current_index = 0

    def _init_vectors(self):
        pass

    def next_point(self, vector, point):
        new_point = Point(round(vector[0] * self._scale + point.x, self._round_dec), round(vector[1] * self._scale + point.y, self._round_dec))
        nearest = min(
            self._nodes,
            key=lambda node: self.distance(node, new_point))
        if self.distance(new_point, nearest) < self._scale / 10:
            return nearest
        else:
            return new_point

    def _index_edges(self):
        super()._index_edges()
        self._index_edges_2d()

    def _index_edges_2d(self):
        pass

    def _create(self):
        pass


class SquareGrid(RegularGrid):
    def _init_vectors(self):
        self.first = (0, 1)
        self.second = (1, 0)

    def _index_edges_2d(self):
        for edge in self._edges:
            index = self._edges[edge]['constr_No']
            i = index // (self._m * 2)
            j = index % (self._m * 2)
            self._edges[edge]['2d'] = (i, j)

    def _create(self):
        for _ in range(self._n):
            current = self._start
            self._start = self.next_point(self.first, self._start)
            for _ in range(self._m):
                u = self.next_point(self.first, current)
                v = self.next_point(self.second, current)
                self.add_edge(current, u)
                self.add_edge(current, v)
                current = v


class HexagonGrid(RegularGrid):
    def _init_vectors(self):
        self.first = (0.5, -math.sqrt(3) * 0.5)
        self.second = (0.5, math.sqrt(3) * 0.5)
        self.third = (1, 0)

    def _index_edges_2d(self):
        for edge in self._edges:
            index = self._edges[edge]['constr_No']
            i = index // (self._m * 3)
            j = index % (self._m * 3)
            self._edges[edge]['2d'] = (i, j)

    def _create(self):
        for _ in range(self._n):
            current = self._start
            self._start = self.next_point((0, math.sqrt(3)), self._start)
            for _ in range(self._m):
                u = self.next_point(self.first, current)
                v = self.next_point(self.second, current)
                w = self.next_point(self.third, v)
                self.add_edge(current, u)
                self.add_edge(current, v)
                self.add_edge(v, w)
                current = w


if __name__ == '__main__':
    layout = HexagonGrid(4, 6, 1)
    layout.draw_pdf('square.pdf')
