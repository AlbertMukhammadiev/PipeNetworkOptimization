from pandas import read_csv
from collections import namedtuple


FileNames = namedtuple('FileNames', 'nodes edges costs')
EdgeProps = namedtuple('EdgeProps', 'label length diameter cost flow_rate actual_flow')
PipeProps = namedtuple('PipeProps', 'diameter cost')
NodeProps = namedtuple('NodeProps', 'pos type demand')


class ReaderCSV:
    def __init__(self, fnames):
        self._fnames = fnames

    def edges_scheme(self):
        edges = read_csv(self._fnames.edges)
        scheme = dict()
        for item in edges.itertuples():
            scheme[eval(item.link)] = EdgeProps(
                label=item.label,
                length=item.length,
                diameter=0,
                cost=0,
                flow_rate=0,
                actual_flow=0,
            )
        return scheme

    def cost_scheme(self):
        cost_data = read_csv(self._fnames.costs)
        key_len = len(bin(len(cost_data) - 1)) - 2
        scheme = dict()
        for item in cost_data.itertuples():
            key = bin(item.Index)[2:].zfill(key_len)
            scheme[key] = PipeProps(
                diameter=item.diameter,
                cost=item.cost,
            )
        return scheme

    def nodes_scheme(self):
        nodes = read_csv(self._fnames.nodes)
        scheme = dict()
        for item in nodes.itertuples():
            scheme[item.label] = NodeProps(pos=eval(item.pos), type=item.type, demand=item.demand)
        return scheme