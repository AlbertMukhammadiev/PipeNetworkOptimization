import random
from deap import base, creator, tools


class GA:
    def __init__(self):
        random.seed(32)
        self._network = None
        self._log = {
            'min': [],
            'max': [],
            'avg': [],
        }

    def _init_ga_properties(self):
        self._individual_size = self._network.nbits_required()
        self._n_generations = 100
        self._n_individuals = 100
        # The probability for mutating an individual
        self._mutation_probability = 0.1
        # The probability with which two individuals are crossed
        self._crossover_probability = 0.5

    @staticmethod
    def individual_2d(n, m):
        return [random.choices([0, 1], k=m) for _ in range(n)]

    @staticmethod
    def crossover_2point_2d(ind1, ind2):
        n = min(len(ind1), len(ind2))
        m = min(len(ind1[0]), len(ind2[0]))
        cx_points = random.choices(range(m), k=2)
        print(cx_points)
        cy_points = random.choices(range(n), k=2)
        print(cy_points)
        cx_point1, cx_point2 = min(cx_points), max(cx_points) + 1
        cy_point1, cy_point2 = min(cy_points), max(cy_points) + 1
        for i in range(cy_point1, cy_point2):
            ind1[i][cx_point1:cx_point2], ind2[i][cx_point1:cx_point2] \
                = ind2[i][cx_point1:cx_point2], ind1[i][cx_point1:cx_point2]

        return ind1, ind2

    @property
    def network(self):
        return 'some network'

    @network.setter
    def network(self, value):
        self._network = value
        self._init_ga_properties()

    @property
    def mutation_probability(self):
        return self._mutation_probability

    @mutation_probability.setter
    def mutation_probability(self, value):
        if value >= 1.0:
            self._mutation_probability = 0.7
        elif value <= 0:
            self._mutation_probability = 0.01
        else:
            self._mutation_probability = value

    @property
    def crossover_probability(self):
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value):
        if value >= 1.0:
            self._crossover_probability = 0.7
        elif value <= 0:
            self._crossover_probability = 0.01
        else:
            self._crossover_probability = value

    @property
    def n_individuals(self):
        return self._n_individuals

    @n_individuals.setter
    def n_individuals(self, value):
        if value > 1000:
            self._n_individuals = 1000
        elif value < 100:
            self._n_individuals = 100
        else:
            self._n_individuals = value

    @property
    def n_generations(self):
        return self._n_generations

    @n_generations.setter
    def n_generations(self, value):
        if value > 1000:
            self._n_generations = 1000
        elif value < 100:
            self._n_generations = 100
        else:
            self._n_generations = value

    def main(self):
        self._configure_algorithm()
        population = self.toolbox.population(n=self._n_individuals)
        print("Start of evolution")
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for individual, fit in zip(population, fitnesses):
            individual.fitness.values = fit

        print(f'  Evaluated {len(population)} individuals')

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in population]

        generation_No = 0
        while generation_No < self._n_generations:
            # self.save_best(pop, g)
            # self.save_worst(pop, g)
            generation_No = generation_No + 1
            print(f'-- Generation {generation_No} --')

            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with crossover probability
                if random.random() < self.crossover_probability:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with mutation probability
                if random.random() < self.mutation_probability:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for individual, fit in zip(invalid_ind, fitnesses):
                individual.fitness.values = fit

            print(f'  Evaluated {len(invalid_ind)} individuals')

            # The population is entirely replaced by the offspring
            population[:] = offspring
            self._update_log(population)

        print('-- End of (successful) evolution --')
        best_ind = tools.selBest(population, 1)[0]
        print(best_ind)
        self._network.design = best_ind
        self._network.draw_pdf('best.pdf')

        print(best_ind.fitness.values[0])
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    def _configure_algorithm(self):
        def evalOneMax(individual):
            self._network.bit_representation = individual
            return self._network.total_cost(),

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.choice, (0, 1))
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self._individual_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evalOneMax)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _update_log(self, population):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        self._log['min'].append(min(fits))
        self._log['max'].append(max(fits))
        self._log['avg'].append(mean)

    def save_best(self, population, generation_No):
        best_ind = tools.selBest(population, 1)[0]
        self._network.bit_representation = best_ind
        self._network.draw_pdf(f'best_in_{generation_No}_generation')

    def save_worst(self, population, generation_No):
        worst_ind = tools.selWorst(population, 1)[0]
        self._network.bit_representation = worst_ind
        self._network.draw_pdf(f'worst_in_{generation_No}_generation')


if __name__ == "__main__":

    # pass

    from properties import PipeProps
    from network import NetworkGA
    from layouts import SquareLayout, HexagonLayout
    from ga import GA


    sinks = {
        (9, 9): -120,
    }

    sources = {
        (0, 0): 10,
        (0, 4): 10,
        (0, 9): 20,
        (4, 1): 20,
        (5, 5): 10,
        (4, 8): 20,
        (8, 6): 20,
        (9, 1): 10,
    }

    cost_model = [PipeProps(diameter=0.0, cost=0.0), PipeProps(diameter=80.0, cost=23.0),
                  PipeProps(diameter=100.0, cost=32.0), PipeProps(diameter=120.0, cost=50.0),
                  PipeProps(diameter=140.0, cost=60.0), PipeProps(diameter=160.0, cost=90.0),
                  PipeProps(diameter=180.0, cost=130.0), PipeProps(diameter=200.0, cost=170.0),
                  PipeProps(diameter=220.0, cost=300.0), PipeProps(diameter=240.0, cost=340.0),
                  PipeProps(diameter=260.0, cost=390.0), PipeProps(diameter=280.0, cost=430.0),
                  PipeProps(diameter=300.0, cost=470.0), PipeProps(diameter=320.0, cost=500.0)]

    network = NetworkGA(sinks, sources)
    network.change_layout(HexagonLayout(10, 10))
    network.cost_model = cost_model

    ga = GA()
    ga.network = network
    ga.n_generations = 300
    ga.n_individuals = 1000
    ga.main()