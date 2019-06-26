import math
import random
import numpy as np
import networkx as nx

import helper


class Builder:
    def __init__(self, layout, cost_model):
        self._layout = layout
        self._init_cost_model(cost_model)
        self._init_sources()
        self._init_sinks()

    def _init_cost_model(self, cost_model):
        if 0 not in [props['diam'] for props in cost_model]:
            cost_model.append(dict(diam=0, cost=0))
        self._cost_model = {i: props for i, props in enumerate(cost_model)}

    def _init_sinks(self):
        self._sinks = [u for u in self._layout.nodes if self._is_sink(u)]

    def _init_sources(self):
        self._sources = [u for u in self._layout.nodes if self._is_source(u)]

    @staticmethod
    def flow_rate(diameter: float, velocity: float):
        return 1 / 4 * math.pi * diameter ** 2 * velocity / 1000

    @property
    def current_design(self):
        design = self._layout.copy()
        unused = [(u, v) for u, v, props in design.edges(data=True) if not props['diam']]
        design.remove_edges_from(unused)
        return design

    def pipe_by_flow(self, flow):
        velocity = 1
        diam = math.sqrt(4 * flow / math.pi / velocity * 1000)
        print(diam)
        pipes = self._cost_model.values()
        suitable_props = filter(lambda x: x['diam'] > diam, pipes)
        cheapest = min(suitable_props, key=lambda x: x['cost'])
        return cheapest

    def draw_design(self, path):
        design = self._layout.copy()
        unused = [(u, v) for u, v, props in design.edges(data=True) if not props['diam']]
        design.remove_edges_from(unused)
        design.draw_pdf(path, {'cost': self.development_cost()})

    def development_cost(self):
        return self.preliminary_cost() + self.penalty_cost()

    def preliminary_cost(self):
        costs = [props['cost'] for _, _, props in self._layout.edges(data=True)]
        lengths = [props['len'] for _, _, props in self._layout.edges(data=True)]
        return sum(np.array(costs) * np.array(lengths))

    def penalty_cost(self):
        return 0

    def _calculate_flows(self):
        current_design = nx.Graph()
        current_design.add_nodes_from(self._layout.nodes)
        existing_edges = [(u, v) for u, v, props in self._layout.edges(data=True) if props['diam']]
        current_design.add_edges_from(existing_edges)
        self._n_isolated_sources = 0
        for source in self._sources:
            for sink in self._sinks:
                try:
                    traces = nx.shortest_path(current_design, source, sink)
                    for u, v in zip(traces, traces[1:]):
                        self._layout.edges[u, v]['flow_rate'] += self._layout.nodes[source]['demand']
                except nx.exception.NetworkXNoPath:
                    self._n_isolated_sources += 1

    def _is_sink(self, node):
        props = self._layout.nodes[node]
        return props['demand'] < 0

    def _is_source(self, node):
        props = self._layout.nodes[node]
        return props['demand'] > 0


class SimpleBuilder(Builder):
    def __init__(self, layout, cost_model):
        super().__init__(layout, cost_model)
        self._validate_all_edges()
        self._calculate_flows()
        self._install_appropriate_diameters()

    def _validate_all_edges(self):
        diams = {edge: 1 for edge in self._layout.edges}
        nx.set_edge_attributes(self._layout, diams, name='diam')

    def _install_appropriate_diameters(self):
        for u, v, props in self._layout.edges(data=True):
            pipe = self.pipe_by_flow(props['flow_rate'])
            props.update(pipe)


class BuilderByConfig(Builder):
    def __init__(self, layout, cost_model):
        super().__init__(layout, cost_model)
        self._init_max_possible_cost()

    def _init_cost_model(self, cost_model):
        if 0 not in [props['diam'] for props in cost_model]:
            cost_model.append(dict(diam=0, cost=0))
        keys = helper.binaries(len(cost_model))
        values = sorted(cost_model, key=lambda pipe: pipe['diam'])
        values.extend((values[-1] for _ in range(len(keys) - len(values))))
        self._cost_model = dict(zip(keys, values))

    def _init_max_possible_cost(self):
        self._most_expensive = max([props['cost'] for props in self._cost_model.values()])
        lengths = [props['len'] for u, v, props in self._layout.edges(data=True)]
        self._max_possible_cost = sum(self._most_expensive * np.array(lengths))

    @property
    def max_possible_cost(self):
        return self._max_possible_cost

    @property
    def config_shape(self):
        return len(next(iter(self._cost_model))) * self._layout.number_of_edges()

    def development_cost(self) -> float:
        if self._n_isolated_sources > 0:
            return self._max_possible_cost * self._n_isolated_sources
        else:
            return super().development_cost()

    def penalty_cost(self):
        lengths = [props['len'] for _, _, props in self._layout.edges(data=True)]
        redevelopment_cost = []
        for u, v, eprops in self._layout.edges(data=True):
            try:
                rate = math.floor(eprops['flow_rate'] / eprops['max_flow_rate'])
            except ZeroDivisionError:
                rate = 0
            redevelopment_cost.append(rate * self._most_expensive)
        return sum(np.array(redevelopment_cost) * np.array(lengths))

    # 1D indexing
    # TODO indexing
    def redesign(self, configuration):
        k = len(next(iter(self._cost_model)))
        for u, v, eprops in self._layout.edges(data=True):
            index = eprops['indexBFS']
            key = configuration[index * k:(index + 1) * k]
            pipe_props = self._cost_model[''.join(map(str, key))]

            self._layout.edges[u, v]['diam'] = pipe_props['diam']
            self._layout.edges[u, v]['cost'] = pipe_props['cost']
            self._layout.edges[u, v]['flow_rate'] = 0
            self._layout.edges[u, v]['max_flow_rate'] = Builder.flow_rate(pipe_props['diam'], 1)
            self._layout.edges[u, v]['gene'] = key
        self._calculate_flows()


        # class GA2d(GA):
        #     @staticmethod
        #     def mutation(individual, indpb):
        #         for i in range(len(individual)):
        #             for j in range(len(individual[i])):
        #                 if random.random() < indpb:
        #                     individual[i][j] = type(individual[i][j])(not individual[i][j])
        #
        #         return individual,
        #
        #     @staticmethod
        #     def individual(n, m):
        #         return [random.choices([0, 1], k=m) for _ in range(n)]
        #
        #     @staticmethod
        #     def crossover_2point(ind1, ind2):
        #         n = min(len(ind1), len(ind2))
        #         m = min(len(ind1[0]), len(ind2[0]))
        #         cx_points = random.choices(range(m), k=2)
        #         cy_points = random.choices(range(n), k=2)
        #         cx_point1, cx_point2 = min(cx_points), max(cx_points) + 1
        #         cy_point1, cy_point2 = min(cy_points), max(cy_points) + 1
        #         for i in range(cy_point1, cy_point2):
        #             ind1[i][cx_point1:cx_point2], ind2[i][cx_point1:cx_point2] \
        #                 = ind2[i][cx_point1:cx_point2], ind1[i][cx_point1:cx_point2]
        #
        #         return ind1, ind2
        #
        #     def attr0(self, k):
        #         return [0 for _ in range(k)]
        #
        #     def _configure_algorithm(self):
        #         creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        #         creator.create("Individual", list, fitness=creator.FitnessMin)
        #         n, m = self._individual_shape
        #         self.toolbox = base.Toolbox()
        #         self.toolbox.register("attr0", self.attr0, k=m)
        #         self.toolbox.register("individual0", tools.initRepeat, creator.Individual,
        #                               self.toolbox.attr0, n)
        #         self.toolbox.register("population0", tools.initRepeat, list, self.toolbox.individual0)
        #
        #         self.toolbox.register("attr_bool", random.choices, (0, 1), k=m)
        #         self.toolbox.register("individual", tools.initRepeat, creator.Individual,
        #                               self.toolbox.attr_bool, n)
        #         self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        #         self.toolbox.register("evaluate", self.evaluate)
        #         self.toolbox.register("mate", GA2d.crossover_2point)
        #         self.toolbox.register("mutate", GA2d.mutation, indpb=0.05)
        #         self.toolbox.register("select", tools.selTournament, tournsize=3)
        #
        # if __name__ == "__main__":
        #     pass


# class NetworkGA2D(NetworkGA):
#     def _init_edge_by_index(self):
#         self._edge_by_index = dict()
#         for edge, props in self._edges.items():
#             self._edge_by_index[props['cindex2d']] = edge
#
#         n = max(i for i, j in self._edge_by_index.keys()) + 1
#         m = max(j for i, j in self._edge_by_index.keys()) + 1
#         self._chromosome_shape = (n, m * self._b_len)
#
#     def redesign(self):
#         n, m = self.chromosome_shape
#         for i in range(n):
#             for j in range(m // self._b_len):
#                 bits = self._chromosome[i][j * self._b_len:(j + 1) * self._b_len]
#                 self._redesign_edge(index=(i, j), gene=bits)
#         self._calculate_flows()
#
#     def _drawing_configurations(self):
#         pos, kwargs = super()._drawing_configurations()
#         for edge in self._design.edges:
#             props = self._edges[edge]
#             label = f'\ncindex2d: {props["cindex2d"]}'
#             kwargs['edge_labels'][edge] += label
#         return pos, kwargs

if __name__ == '__main__':
    pass