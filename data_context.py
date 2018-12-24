from networkx import read_gexf, write_gexf
from collections import namedtuple
import json


PipeProps = namedtuple('PipeProps', 'diameter cost')


class DataContext:
    def __init__(self, path):
        self._directory = path

    def cost_model(self):
        path = self._directory + 'cost_model.json'
        with open(path) as fin:
            model = json.load(fin)
        return [PipeProps(diameter=float(key), cost=float(value)) for key, value in model.items()]

    def initial_graph(self):
        path = self._directory + 'initial.gexf'
        return read_gexf(path, node_type=int)

    def initial_nodes(self):
        path = self._directory + 'initial.gexf'
        graph = read_gexf(path, node_type=int)
        return graph.nodes(data=True)

    def initial_edges(self):
        path = self._directory + 'initial.gexf'
        graph = read_gexf(path, node_type=int)
        return graph.edges(data=True)

    def save(self, graph):
        write_gexf(graph, self._directory + 'test.gexf')