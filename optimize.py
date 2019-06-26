import json
import math
import random
import time

from deap import base
from deap import creator
from deap import tools

from builder import BuilderByConfig


class Optimizer:
    def __init__(self, layout, cost_model):
        self._init_builder(layout, cost_model)
        self._init_log(layout)
        self._init_configurations()
        random.seed(time.time())

    def _init_builder(self, layout, cost_model):
        self._builder = BuilderByConfig(layout, cost_model)
        self._config_shape = self._builder.config_shape

    def _init_configurations(self):
        pass

    def _init_log(self, layout):
        self._log_name = f'_{str(self)}_{str(layout)}_.json'
        self._log = {
            # 'best': ([], self._builder.max_possible_cost)
            'best': ([], 100000000000000)
        }

    def evaluate(self, configuration):
        self._builder.redesign(configuration)
        return self._builder.development_cost()

    def optimize(self):
        pass

    def _compare_with_best(self, configuration, cost):
        if cost < self._log['best'][1]:
            self._log['best'] = (configuration, cost)

    def save_log(self, time):
        self._log['time'] = time
        with open(self._log_name, 'w') as f:
            json.dump(self._log, f, indent=4)


class SA(Optimizer):
    def _init_configurations(self):
        configurations = {
            'T': 500,
            't': 1,
            'alpha': 0.99,
            'L': 1000,
            'costs': [],
        }
        self._log.update(configurations)

    def initial_configuration(self):
        return random.choices((0, 1), k=self._config_shape)

    def energy(self, configuration):
        return super().evaluate(configuration)

    def optimize(self):
        def rearrangement(configuration):
            pos = random.randint(0, len(configuration) - 1)
            configuration[pos] = type(configuration[pos])(not configuration[pos])

        def rearrangement2(configuration):
            for i in range(len(configuration)):
                if random.random() < 0.05:
                    configuration[i] = type(configuration[i])(not configuration[i])

        start = time.time()
        config = self.initial_configuration()
        e = self.energy(config)
        T, t = self._log['T'], self._log['t']
        alpha = self._log['alpha']
        L = self._log['L']
        while T > t:
            print(T)
            for _ in range(L):
                next_config = config.copy()
                rearrangement(next_config)
                next_e = self.energy(next_config)
                self._update_log(next_config, next_e, T)
                de = next_e - e
                if de > 0.0 and random.random() <= math.exp(-de / T) or \
                        de < 0.0:
                    config = next_config
                    e = next_e
            T = alpha * T

        end = time.time()
        self.save_log(end - start)

    def _update_log(self, configuration, cost, t):
        self._log['costs'].append((t, cost))
        self._compare_with_best(configuration, cost)

    def __str__(self):
        return 'sa'


class GA(Optimizer):
    def _init_log(self, layout):
        super()._init_log(layout)
        self._log.update({'max': [], 'min': [], 'avg': []})

    def _init_configurations(self):
        configurations = {
            'n_generations': 1000,
            'n_individuals': 1500,
            'mutation_probability': 0.1,
            'crossover_probability': 0.5
        }
        self._log.update(configurations)

    def _configure_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.choice, (0, 1))

        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self._config_shape)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual):
        return super().evaluate(individual),

    def optimize(self):
        self._configure_algorithm()
        start = time.time()
        population = self.toolbox.population(n=self._log['n_individuals'])
        # individual0 = self.toolbox.population0(n=1)[0]
        # population[0] = individual0
        print('Start of evolution')

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = fitness

        print(f'  Evaluated {len(population)} individuals')

        # Extracting all the fitnesses of
        fits = [individual.fitness.values[0] for individual in population]

        for i in range(1, self._log['n_generations']):
            print(f'-- Generation {i} --')
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with crossover probability
                if random.random() < self._log['crossover_probability']:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with mutation probability
                if random.random() < self._log['mutation_probability']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for individual, fitness in zip(invalid_ind, fitnesses):
                individual.fitness.values = fitness

            print(f'  Evaluated {len(invalid_ind)} individuals')

            # The population is entirely replaced by the offspring
            population[:] = offspring
            self._update_log(population)
        print('-- End of (successful) evolution --')
        end = time.time()
        self.save_log(end - start)

    def _update_log(self, population):
        def std(fits):
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            print("  Std %s" % std)

        def _update_log(self, configuration, cost, i):
            self._log[i] = cost

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        # print(f'  Min {min(fits)}')
        # print(f'  Max {max(fits)}')
        # print(f'  Avg {sum(fits) / len(population)}')
        # std(fits)
        self._log['min'].append(min(fits))
        self._log['max'].append(max(fits))
        self._log['avg'].append(sum(fits) / len(population))
        a = tools.selBest(population, 1)
        best_ind = a[0]
        self._compare_with_best(best_ind, min(fits))

    def __str__(self):
        return 'ga'
