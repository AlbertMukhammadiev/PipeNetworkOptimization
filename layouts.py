import math
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product, combinations
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

import helper

from functools import partial
round2 = partial(round, ndigits=2)


class Layout(nx.Graph):
    def __init__(self):
        super().__init__()
        self._indexing_strategy = None
        self._node_by_index = dict()
        self._edges_by_index = dict()

    @property
    def indexing_strategy(self):
        return self._indexing_strategy

    @indexing_strategy.setter
    def indexing_strategy(self, value):
        if self.indexing_strategy == value:
            return

        self._edges_by_index.clear()
        if value == 'bfs':
            self._index_edges_bfs()
            for u, v, props in self.edges(data=True):
                index = props['indexBFS']
                self._edges_by_index[index] = (u, v)
        elif value == 'construction':
            for edge, props in self.edges(data=True):
                index = props['cindex']
                self._edges_by_index[index] = edge

    @property
    def edges_by_index(self):
        return self._edges_by_index

    def add_terminal(self, pos, demand):
        self.add_node(pos, pos=pos, demand=demand)

    def add_nonterminal(self, pos):
        self.add_node(pos, pos=pos, demand=0)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        length = self._weight(u_of_edge, v_of_edge)
        cindex = self.number_of_edges()
        super().add_edge(u_of_edge, v_of_edge,
                         len=length, cindex=cindex, flow_rate=0,
                         **attr)
        self.indexing_strategy = None

    def _weight(self, u, v):
        u_pos = self.nodes[u]['pos']
        v_pos = self.nodes[v]['pos']
        return helper.distance(u_pos, v_pos)

    def _index_edges_bfs(self):
        start_node = sorted(self.nodes, reverse=True).pop()
        path_bfs = nx.edge_bfs(self, start_node)
        for i, edge in enumerate(path_bfs):
            self.edges[edge]['indexBFS'] = i

    def _is_terminal(self, u):
        return self.nodes[u]['demand'] != 0

    def _is_nonterminal(self, u):
        return not self._is_terminal(u)

    def draw(self):
        positions = nx.get_node_attributes(self, 'pos')
        plt.axis('equal')
        plt.axis('off')
        nx.draw_networkx_nodes(self, positions)
        nx.draw_networkx_edges(self, positions)
        plt.show()
        plt.close()

    def draw_in_detail(self):
        pos = nx.get_node_attributes(self, 'pos')
        kwargs = self._drawing_configurations()
        plt.axis('equal')
        plt.axis('off')
        nx.draw_networkx_nodes(self, pos, **kwargs)
        nx.draw_networkx_labels(self, pos, **kwargs)
        nx.draw_networkx_edges(self, pos, **kwargs)
        nx.draw_networkx_edge_labels(self, pos, **kwargs)
        plt.show()
        plt.close()

    def draw_pdf(self, path, info=None):
        pos = nx.get_node_attributes(self, 'pos')
        kwargs = self._drawing_configurations()
        plt.axis('equal')
        plt.axis('off')
        nx.draw_networkx_nodes(self, pos, **kwargs)
        nx.draw_networkx_labels(self, pos, **kwargs)
        nx.draw_networkx_edges(self, pos, **kwargs)
        nx.draw_networkx_edge_labels(self, pos, **kwargs)
        if info:
            plt.title(f'cost: {info["cost"]}')
        plt.savefig(path, format='pdf')
        plt.close()

    def _drawing_configurations(self):
        kwargs = {
            'node_color': [],
            'node_size': 80,
            'node_shape': 's',
            'alpha': 1,
            'font_size': 1,
            'rotate': False,
            'width': 0.2,
            'labels': {},
            'edge_labels': {}}

        for node, props in self.nodes(data=True):
            label = f'{node}\npos: {props["pos"]}\n{props["demand"]}'
            kwargs['labels'][node] = node
            if props['demand'] > 0:
                color = 'lightgreen'
            elif props['demand'] < 0:
                color = 'salmon'
            else:
                color = 'lightgoldenrodyellow'
            kwargs['node_color'].append(color)

        for u, v, props in self.edges(data=True):
            info = []
            for key, value in props.items():
                info.append(f'{key}: {value}')
            label = '\n'.join(info)
            kwargs['edge_labels'][u, v] = label

        return kwargs

    def __str__(self):
        return 'user'


class CompleteLayout(Layout):
    def __init__(self, terminals=None):
        super().__init__()
        if terminals:
            for pos, demand in terminals:
                self.add_terminal(pos=pos, demand=demand)

            for u, v in combinations(self.nodes, 2):
                self.add_edge(u, v)
            self._index_edges_bfs()


    def __str__(self):
        return 'complete'


class DelaunayLayout(Layout):
    def __init__(self, terminals=None):
        super().__init__()
        if terminals:
            for pos, demand in terminals:
                self.add_terminal(pos=pos, demand=demand)

            points = [pos for pos, _ in terminals]
            triangulation = Delaunay(points)
            for vertices in triangulation.simplices:
                path = [points[i] for i in vertices]
                pairs = zip(path, path[1:])
                #TODO incorrect indexing by cindex
                for pair in pairs:
                    self.add_edge(*pair)
            self._index_edges_bfs()



    def __str__(self):
        return 'delaunay'


class StarLayout(Layout):
    def __init__(self, terminals=None, center=None):
        super().__init__()
        if terminals and center:
            self.add_nonterminal(center)
            for pos, demand in terminals:
                self.add_terminal(pos=pos, demand=demand)
            for node in self.nodes:
                self.add_edge(center, node)

    def __str__(self):
        return 'star'


def mst(points):
    graph = CompleteLayout([(p, 0) for p in points])
    b = nx.minimum_spanning_tree(graph, weight='len')
    return b


def delta_mst(points, additional_point):
    cost1 = mst(points).size(weight='len')
    cost2 = mst(points + [additional_point]).size(weight='len')
    return cost1 - cost2


class SteinerLayout(Layout):
    def __init__(self, terminals=None):
        super().__init__()
        if terminals:
            original_points = [pos for pos, _ in terminals]
            steiner_points = []
            candidates = [0]
            while candidates:
                print(len(candidates))
                max_point = (0, 0)
                candidates = [p for p in helper.brute_points(original_points + steiner_points)
                              if delta_mst(original_points + steiner_points, p) > 0]
                cost = 0
                for point in candidates:
                    delta = delta_mst(original_points + steiner_points, point)
                    if delta > cost:
                        max_point = point
                        cost = delta

                if max_point[0] != 0 and max_point[1] != 0:
                    steiner_points.append(max_point)
                layout = mst(original_points + steiner_points)
                for point in steiner_points:
                    if len(list(layout.neighbors(point))) <= 2:
                        steiner_points.remove(point)
                    else:
                        pass

            steiner_tree = mst(original_points + steiner_points)
            for pos, demand in terminals:
                self.add_terminal(pos=pos, demand=demand)
            for point in steiner_points:
                self.add_nonterminal(pos=point)
            for edge in steiner_tree.edges:
                self.add_edge(*edge)

    def __str__(self):
        return 'steiner'


class MultilevelStarLayout(Layout):
    def __init__(self, terminals=None):
        super().__init__()
        if terminals:
            points = [pos for pos, _ in terminals]
            kmeans = KMeans(n_clusters=5).fit(points)
            centers = [tuple(center) for center in kmeans.cluster_centers_]
            clusters = [[] for _ in range(kmeans.n_clusters)]
            for i, terminal in enumerate(terminals):
                i_cluster = kmeans.labels_[i]
                clusters[i_cluster].append(terminal)

            stars = [StarLayout(cluster, center) for cluster, center in zip(clusters, centers)]
            steiner_tree = SteinerLayout([(center, 0) for center in centers])
            result = nx.compose_all(stars + [steiner_tree])

            self.add_nodes_from(result.nodes(data=True))
            self.add_edges_from(result.edges(data=True))

    def __str__(self):
        return 'multilvlstar'


class RegularGrid(Layout):
    def __init__(self, terminals=None, m=None, n=None):
        super().__init__()
        if terminals:
            self._create_grid([pos for pos, _ in terminals], m, n)
            # self._index_edges_2d(m, n)
            self._init_terminals(terminals)
            self._index_edges_bfs()

    def _index_edges_2d(self, m, n):
        pass

    def _create_grid(self, points, m, n):
        pass

    def _nearest_to(self, pos):
        positions = nx.get_node_attributes(self, 'pos')
        return min(
            positions,
            key=lambda node: helper.distance(positions[node], pos))

    def next_point(self, vector, point):
        new_point = (round2(vector[0] + point[0]), round2(vector[1] + point[1]))
        nearest = self._nearest_to(new_point)
        if helper.distance(new_point, self.nodes[nearest]['pos']) < 0.1:
            return nearest
        else:
            self.add_nonterminal(new_point)
            return new_point

    def _init_terminals(self, terminals):
        for pos, demand in terminals:
            node = self._nearest_to(pos)

            self.nodes[node]['pos'] = pos
            self.nodes[node]['demand'] = demand
            # self.add_terminal(pos=pos, demand=demand)


class SquareGrid(RegularGrid):
    def _index_edges_2d(self, m, n):
        for u, v, props in self.edges(data=True):
            i = props['cindex'] // (m * 2)
            j = props['cindex'] % (m * 2)
            props['cindex2d'] = (i, j)

    def _create_grid(self, points, m, n):
        xmax = (max(points, key=lambda p: p[0]))[0]
        xmin = (min(points, key=lambda p: p[0]))[0]
        ymax = (max(points, key=lambda p: p[1]))[1]
        ymin = (min(points, key=lambda p: p[1]))[1]

        dx = (xmax - xmin) / m
        dy = (ymax - ymin) / n

        first = (0, dy)
        second = (dx, 0)

        start = (xmin, ymin)
        self.add_nonterminal(start)
        for _ in range(n):
            current = start
            start = self.next_point(first, start)
            for _ in range(m):
                u = self.next_point(first, current)
                v = self.next_point(second, current)
                self.add_edge(current, u)
                self.add_edge(current, v)
                current = v

    def __str__(self):
        return 'square'


class HexagonGrid(RegularGrid):
    @staticmethod
    def b(p):
        return p[1] - math.sqrt(3) / 2 * p[0]

    def _index_edges_2d(self, m, n):
        for u, v, props in self.edges(data=True):
            i = props['cindex'] // (m * 3)
            j = props['cindex'] % (m * 3)
            props['cindex2d'] = (i, j)

    def _create_grid(self, points, m, n):
        xmax = (max(points, key=lambda p: p[0]))[0]
        xmin = (min(points, key=lambda p: p[0]))[0]
        bmax = HexagonGrid.b((max(points, key=lambda p: HexagonGrid.b(p))))
        bmin = HexagonGrid.b((min(points, key=lambda p: HexagonGrid.b(p))))
        print(f'bmax {bmax}')
        print(f'bmin {bmin}')
        print(f'xmax {xmax}')
        print(f'xmin {xmin}')

        ymin = math.sqrt(3) / 2 * xmin + bmin
        print(f'ymin {ymin}')

        dx = (xmax - xmin) / m
        dy = (bmax - bmin) / n
        print(f'dx {dx}')
        print(f'dy {dy}')

        first = (dx / 3, - 0.5 * dy)
        second = (dx / 3, 0.5 * dy)
        third = (dx * 2 / 3, 0)

        start = (xmin, ymin)
        self.add_nonterminal(start)
        for _ in range(n):
            current = start
            start = self.next_point((0, dy), start)
            for _ in range(m):
                u = self.next_point(first, current)
                v = self.next_point(second, current)
                w = self.next_point(third, v)
                self.add_edge(current, u)
                self.add_edge(current, v)
                self.add_edge(v, w)
                current = w
        for node in self.nodes:
            if len(list(self.neighbors(node))) == 0:
                self.remove_node(node)
                break



    def __str__(self):
        return 'hexagon'


def create_layout1():
    layout = Layout()
    layout.add_node(1, pos=(0, 0), demand=10)
    layout.add_node(2, pos=(1, 0), demand=20)
    layout.add_node(3, pos=(0, 1), demand=10)
    layout.add_node(4, pos=(2, 0), demand=20)
    layout.add_node(5, pos=(1, 1), demand=10)
    layout.add_node(6, pos=(0, 2), demand=20)
    layout.add_node(7, pos=(2, 1), demand=10)
    layout.add_node(8, pos=(1, 2), demand=20)
    layout.add_node(9, pos=(2, 2), demand=-120)

    layout.add_edge(1, 2)
    layout.add_edge(1, 3)
    layout.add_edge(2, 4)
    layout.add_edge(2, 5)
    layout.add_edge(3, 5)
    layout.add_edge(3, 6)
    layout.add_edge(4, 7)
    layout.add_edge(5, 7)
    layout.add_edge(5, 8)
    layout.add_edge(6, 8)
    layout.add_edge(7, 9)
    layout.add_edge(8, 9)
    return layout


def create_layout2():
    layout = Layout()
    layout.add_node(1, pos=(7.5, 2), demand=20)
    layout.add_node(2, pos=(19.8, 3.2), demand=20)
    layout.add_node(3, pos=(34.6, 4.6), demand=15)
    layout.add_node(4, pos=(1.6, 7), demand=30)
    layout.add_node(5, pos=(27.5, 9.5), demand=10)
    layout.add_node(6, pos=(14.6, 11.7), demand=10)
    layout.add_node(7, pos=(34.6, 16), demand=20)
    layout.add_node(8, pos=(2.8, 18), demand=5)
    layout.add_node(9, pos=(8.9, 17.6), demand=10)
    layout.add_node(10, pos=(26.2, 18.5), demand=15)
    layout.add_node(11, pos=(20.1, 22.4), demand=20)
    layout.add_node(12, pos=(30.7, 24.6), demand=25)
    layout.add_node(13, pos=(40.5, 23.8), demand=10)
    layout.add_node(14, pos=(9.9, 28.9), demand=5)
    layout.add_node(15, pos=(3.9, 29.9), demand=10)
    layout.add_node(16, pos=(14.2, 32.2), demand=-1111)
    layout.add_node(17, pos=(33.3, 31.2), demand=20)
    layout.add_node(18, pos=(21.1, 36.1), demand=30)
    layout.add_node(19, pos=(7.9, 37), demand=10)
    layout.add_node(20, pos=(16.3, 43.2), demand=12)

    layout.add_edge(1, 2)
    layout.add_edge(4, 1)
    layout.add_edge(1, 6)
    layout.add_edge(2, 3)
    layout.add_edge(2, 5)
    layout.add_edge(6, 2)
    layout.add_edge(5, 3)
    layout.add_edge(7, 3)
    layout.add_edge(4, 8)
    layout.add_edge(4, 9)
    layout.add_edge(5, 7)
    layout.add_edge(10, 5)
    layout.add_edge(9, 6)
    layout.add_edge(6, 10)
    layout.add_edge(12, 7)
    layout.add_edge(7, 13)
    layout.add_edge(8, 9)
    layout.add_edge(8, 15)
    layout.add_edge(9, 11)
    layout.add_edge(9, 14)
    layout.add_edge(11, 10)
    layout.add_edge(10, 12)
    layout.add_edge(11, 12)
    layout.add_edge(16, 11)
    layout.add_edge(11, 18)
    layout.add_edge(12, 13)
    layout.add_edge(12, 17)
    layout.add_edge(17, 13)
    layout.add_edge(15, 14)
    layout.add_edge(14, 16)
    layout.add_edge(19, 14)
    layout.add_edge(15, 19)
    layout.add_edge(16, 18)
    layout.add_edge(19, 16)
    layout.add_edge(18, 17)
    layout.add_edge(20, 18)
    layout.add_edge(19, 20)
    return layout


def create_Hanoi_layout():
    layout = Layout()
    layout.add_node(
        1,
        pos=(5360.71, 1354.05),
        demand=-111111111111111111111
    )
    layout.add_node(
        2,
        pos=(5021.20, 1582.17),
        demand=890
    )
    layout.add_node(
        3,
        pos=(5025.20, 2585.42),
        demand=850
    )
    layout.add_node(
        4,
        pos=(5874.22, 2588.30),
        demand=130
    )
    layout.add_node(
        5,
        pos=(6873.11, 2588.30),
        demand=725
    )
    layout.add_node(
        6,
        pos=(8103.51, 2585.42),
        demand=1005
    )
    layout.add_node(
        7,
        pos=(8103.51, 3234.67),
        demand=1350
    )
    layout.add_node(
        8,
        pos=(8106.66, 4179.28),
        demand=550
    )
    layout.add_node(
        9,
        pos=(8106.66, 5133.78),
        demand=525
    )
    layout.add_node(
        10,
        pos=(7318.64, 5133.78),
        demand=525
    )
    layout.add_node(
        11,
        pos=(7319.94, 5831.65),
        demand=500
    )
    layout.add_node(
        12,
        pos=(7319.94, 6671.19),
        demand=560
    )
    layout.add_node(
        13,
        pos=(5636.76, 6676.24),
        demand=940
    )
    layout.add_node(
        14,
        pos=(6530.63, 5133.78),
        demand=615
    )
    layout.add_node(
        15,
        pos=(5676.02, 5133.78),
        demand=280
    )
    layout.add_node(
        16,
        pos=(5021.20, 5133.78),
        demand=310
    )
    layout.add_node(
        17,
        pos=(5021.20, 4412.36),
        demand=865
    )
    layout.add_node(
        18,
        pos=(5021.20, 3868.52),
        demand=1345
    )
    layout.add_node(
        19,
        pos=(5021.20, 3191.49),
        demand=60
    )
    layout.add_node(
        20,
        pos=(3587.87, 2588.30),
        demand=1275
    )
    layout.add_node(
        21,
        pos=(3587.87, 1300.84),
        demand=930
    )
    layout.add_node(
        22,
        pos=(3587.87, 901.29),
        demand=485
    )
    layout.add_node(
        23,
        pos=(1978.55, 2588.30),
        demand=1045
    )
    layout.add_node(
        24,
        pos=(1975.58, 4084.35),
        demand=820
    )
    layout.add_node(
        25,
        pos=(1980.46, 5137.63),
        demand=170
    )
    layout.add_node(
        26,
        pos=(3077.46, 5137.63),
        demand=900
    )
    layout.add_node(
        27,
        pos=(3933.52, 5133.78),
        demand=370
    )
    layout.add_node(
        28,
        pos=(846.04, 2588.20),
        demand=290
    )
    layout.add_node(
        29,
        pos=(-552.41, 2588.20),
        demand=360
    )
    layout.add_node(
        30,
        pos=(-552.38, 4369.06),
        demand=360
    )
    layout.add_node(
        31,
        pos=(-549.36, 5137.63),
        demand=105
    )
    layout.add_node(
        32,
        pos=(536.45, 5137.63),
        demand=805
    )

    layout.add_edge(1, 2)
    layout.add_edge(2, 3)
    layout.add_edge(3, 4)
    layout.add_edge(4, 5)
    layout.add_edge(5, 6)
    layout.add_edge(6, 7)
    layout.add_edge(7, 8)
    layout.add_edge(8, 9)
    layout.add_edge(9, 10)
    layout.add_edge(10, 11)
    layout.add_edge(11, 12)
    layout.add_edge(12, 13)
    layout.add_edge(10, 14)
    layout.add_edge(14, 15)
    layout.add_edge(15, 16)
    layout.add_edge(17, 16)
    layout.add_edge(18, 17)
    layout.add_edge(19, 18)
    layout.add_edge(3, 19)
    layout.add_edge(3, 20)
    layout.add_edge(20, 21)
    layout.add_edge(21, 22)
    layout.add_edge(20, 23)
    layout.add_edge(23, 24)
    layout.add_edge(24, 25)
    layout.add_edge(26, 25)
    layout.add_edge(27, 26)
    layout.add_edge(16, 27)
    layout.add_edge(23, 28)
    layout.add_edge(28, 29)
    layout.add_edge(29, 30)
    layout.add_edge(30, 31)
    layout.add_edge(32, 31)
    layout.add_edge(25, 32)
    return layout


if __name__ == '__main__':
    terminals = [
        ((1, 0), 10),
        ((5, 5), 10),
        ((3, 14), 10),
    ]

    la = HexagonGrid(terminals, 3, 6)
    la.draw_pdf('la.pdf')
