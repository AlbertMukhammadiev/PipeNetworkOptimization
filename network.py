import networkx as nx
import math
from pprint import pprint
from matplotlib.pyplot import savefig
from data_context import DataContext

class Network:
    def __init__(self, data_context) -> None:
        self._internal = data_context.initial_graph()
        self._sinks = dict()
        self._sources = dict()
        for node, props in self._internal.nodes(data=True):
            if props['type'] == 'sink':
                self._sinks[node] = props
            elif props['type'] == 'source':
                self._sources[node] = props
        self._max_layout = {(u, v): props for u, v, props in self._internal.edges(data=True)}
        self._cost_model = self._close_scheme(data_context.cost_model())
        self._edges_by_substringNo = self._map_substringNo()

        self._MAX_COST = self._max_cost()
        self._NBITS_4PIPE = (len(bin(len(self._cost_model) - 1)) - 2)
        self._VELOCITY = 1


    def nbits_required(self):
        return len(self._max_layout) * self._NBITS_4PIPE

    def _calculate_flows(self):
        for source in self._sources:
            for sink, sink_props in self._sinks.items():
                path = nx.shortest_path(self._internal, sink, source)
                for u, v in zip(path, path[1:]):
                    self._internal[u][v]['actual_flow'] += sink_props['demand']

    def penalty_cost(self):
        penalty = 0
        for _, _, props in self._internal.edges(data=True):
            rate = props['actual_flow'] / props['flow_rate']
            if rate > 1:
                penalty += props['cost'] * math.ceil(rate) * props['length'] * 10
        return penalty

    def total_cost(self, individual: list) -> float:
        self.redesign(individual)

        ncomponents = nx.algorithms.number_connected_components(self._internal)
        if ncomponents > 1:
            return self._MAX_COST * ncomponents

        total_cost = 0
        for _, _, props in self._internal.edges(data=True):
            total_cost += props['cost'] * props['length']

        self._calculate_flows()
        # self.draw('flows')
        penalty = self.penalty_cost()
        return total_cost + penalty

    def redesign(self, bits: str) -> None:
        self._reset()
        def flow_rate(diam: float, velocity: float):
            # TODO the dependence on units of measurement
            return 1 / 4 * math.pi * (diam) ** 2 * velocity / 1000

        bits = ''.join(str(bit) for bit in bits)
        substrings = [bits[i:i + self._NBITS_4PIPE] for i in range(0, len(bits), self._NBITS_4PIPE)]
        for i, substring in enumerate(substrings):
            pipe_props = self._cost_model[substring]
            if pipe_props.cost:
                edge = self._edges_by_substringNo[i]
                props = dict(
                    label=self._max_layout[edge]['label'],
                    length=self._max_layout[edge]['length'],
                    diameter=pipe_props.diameter,
                    cost=pipe_props.cost,
                    flow_rate=flow_rate(pipe_props.diameter, self._VELOCITY),
                    actual_flow=0,
                )
                self._internal.add_edge(*edge, **props)

    def draw(self, fname: 'filename.png'):
        edge_attributes = dict()
        for u, v, props in self._internal.edges(data=True):
            label = f"diam {props['diameter']}   c {props['cost']}\nfr {round(props['flow_rate'])}   af {props['actual_flow']}\nl {props['length']}"
            edge_attributes[(u, v)] = label

        node_pos = nx.get_node_attributes(self._internal, 'position')
        for node, pos in node_pos.items():
            ppos = eval(pos)
            node_pos[node] = (ppos['x'], ppos['y']) 
        pprint(node_pos)
        nx.draw(self._internal, node_pos, with_labels=True)
        nx.draw_networkx_edge_labels(self._internal, node_pos, edge_labels=edge_attributes)
        savefig(fname + '.png', format='PNG')
        print(f'save internal in {fname}' + '.png')

    def _close_scheme(self, scheme):
        sorted_props = sorted(scheme, key=lambda pipe: pipe.diameter)
        key_len = len(bin(len(scheme) - 1)) - 2
        scheme = dict()
        for num in range(2 ** key_len):
            key = bin(num)[2:].zfill(key_len)
            try:
                scheme[key] = sorted_props[num]
            except IndexError:
                scheme[key] = sorted_props[-1]
            else:
                pass
            finally:
                pass
        return scheme

    def _max_cost(self):
        most_expensive = max([pipe.cost for pipe in self._cost_model.values()])
        weights = [edge['length'] * most_expensive for edge in self._max_layout.values()]
        return sum(weights)

    def _map_substringNo(self):
        sorted_edges = sorted(self._max_layout.items(), key=lambda pair: pair[1]['label'])
        links = [link for link, props in sorted_edges]
        substringsNo = list(range(len(self._max_layout)))
        return dict(zip(substringsNo, links))

    def _reset(self):
        edges = list(self._internal.edges())
        self._internal.remove_edges_from(edges)


if __name__ == '__main__':
    path = 'projects/square_layout/'
    data_context = DataContext(path)
    network = Network(data_context)
    individual = [
        0,1,0,0,     # 1
        0,0,0,0,     # 2
        0,0,1,1,     # 3
        0,0,1,0,     # 4
        0,0,0,1,     # 5
        0,0,0,0,     # 6
        1,1,0,0,     # 7
        0,1,0,1,     # 8
        0,0,0,0,     # 9
        1,0,1,0,     # 10
        0,0,1,1,     # 11
        0,0,1,0]     # 12
    
    network.redesign(individual)
    network.draw('test')

    # print(network.total_cost(individual))
    # network.save_gexf('qwerty')
    
    # for node in network._internal.nodes(data=True):
    #     print(node)

    # from networkx.readwrite.gexf import read_gexf, write_gexf

    # G = nx.path_graph(4)
    
    # G.nodes[0]['pos'] = {'x': 1, 'y': 3}
    # G.nodes[0]['type'] = 'sink'
    # G.nodes[0]['size'] = -123
    
    # for node in G.nodes(data=True):
    #     print(node)
    # print()
    # nx.write_gexf(G, "test.gexf")



    # g = read_gexf('test.gexf', node_type=int)
    # for node in g.nodes(data=True):
    #     print(node)

    # # g = read_gexf('qwerty.gexf', node_type=int)
    # # for node in g.nodes(data=True):
    # #     print(node)

    # # print(network.total_cost(individual))
