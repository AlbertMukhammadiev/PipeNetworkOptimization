import networkx as nx
from collections import namedtuple
import json
import matplotlib.pyplot as plt


PipeProps = namedtuple('PipeProps', 'diameter cost')


class DataContext:
    def __init__(self, path):
        self._directory = path

    def cost_model(self):
        path = self._directory + 'cost_model.json'
        with open(path) as fin:
            model = json.load(fin)
        return [PipeProps(diameter=float(key), cost=float(value)) for key, value in model.items()]

    # def save_cost_model(self, cost_model):
    #     from pandas import Series
    #     series = Series(cost_model)
    #     df = series.to_frame()

    def save_png(self, graph, path):
        edge_attributes = dict()
        for u, v, props in graph.edges(data=True):
            try:
                label = f"diam {props['diameter']}   c {props['cost']}\nfr {round(props['flow_rate'])}   af {props['actual_flow']}\nl {props['length']}"
            except KeyError:
                label = f"length {props['length']}"
            edge_attributes[(u, v)] = label

        node_pos = nx.get_node_attributes(graph, 'position')
        for node, pos in node_pos.items():
            ppos = eval(pos)
            node_pos[node] = (ppos['x'], ppos['y']) 
        
        node_labels = dict((n, d['demand']) for n, d in graph.nodes(data=True))
        nx.draw(graph, node_pos, labels=node_labels, with_labels=True, node_size=1000, node_color='aqua')
        nx.draw_networkx_edge_labels(graph, node_pos, edge_labels=edge_attributes)
        full_path = self._directory + path + '.png'
        plt.savefig(full_path, format='PNG')
        plt.close()

    def initial_graph(self):
        path = self._directory + 'initial.gexf'
        return nx.read_gexf(path, node_type=int)

    def initial_nodes(self):
        path = self._directory + 'initial.gexf'
        graph = nx.read_gexf(path, node_type=int)
        return graph.nodes(data=True)

    def initial_edges(self):
        path = self._directory + 'initial.gexf'
        graph = nx.read_gexf(path, node_type=int)
        return graph.edges(data=True)

    def save(self, graph):
        write_gexf(graph, self._directory + 'test.gexf')