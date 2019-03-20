import random
from deap import base
from deap import creator
from deap import tools


class GA:
    def __init__(self, network):
        self._network = network
        self._log = {'min': [], 'max': [], 'avg': []}
        self._init_ga_properties()

    def _init_ga_properties(self):
        self._n_generations = 100
        self._n_individuals = 100
        self._mutation_probability = 0.1
        self._crossover_probability = 0.5
        self._individual_shape = self._network.chromosome_shape

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

    def evaluate(self, individual):
        self._network.chromosome = individual
        return self._network.development_cost(),

    def run(self):
        self._configure_algorithm()
        population = self.toolbox.population(n=self.n_individuals)
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

        i_generation = 0
        while i_generation < self._n_generations:
            i_generation = i_generation + 1
            print(f'-- Generation {i_generation} --')

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
            for individual, fitness in zip(invalid_ind, fitnesses):
                individual.fitness.values = fitness

            print(f'  Evaluated {len(invalid_ind)} individuals')

            # The population is entirely replaced by the offspring
            population[:] = offspring
            # population[0] = individual0
            self._update_log(population)

        print('-- End of (successful) evolution --')
        best_ind = tools.selBest(population, 1)[0]
        self._draw_individual(best_ind, 'best.pdf')
        print(f'Best individual is {best_ind}, {best_ind.fitness.values}')

    def _configure_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.choice, (0, 1))

        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self._individual_shape)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
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

    def _draw_individual(self, individual, path='individual.pdf'):
        self._network.chromosome = individual
        self._network.draw_pdf(path)


class GA2d(GA):
    @staticmethod
    def mutation(individual, indpb):
        for i in range(len(individual)):
            for j in range(len(individual[i])):
                if random.random() < indpb:
                    individual[i][j] = type(individual[i][j])(not individual[i][j])

        return individual,

    @staticmethod
    def individual(n, m):
        return [random.choices([0, 1], k=m) for _ in range(n)]

    @staticmethod
    def crossover_2point(ind1, ind2):
        n = min(len(ind1), len(ind2))
        m = min(len(ind1[0]), len(ind2[0]))
        cx_points = random.choices(range(m), k=2)
        cy_points = random.choices(range(n), k=2)
        cx_point1, cx_point2 = min(cx_points), max(cx_points) + 1
        cy_point1, cy_point2 = min(cy_points), max(cy_points) + 1
        for i in range(cy_point1, cy_point2):
            ind1[i][cx_point1:cx_point2], ind2[i][cx_point1:cx_point2] \
                = ind2[i][cx_point1:cx_point2], ind1[i][cx_point1:cx_point2]

        return ind1, ind2

    def attr0(self, k):
        return [0 for _ in range(k)]

    def _configure_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        n, m = self._individual_shape
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr0", self.attr0, k=m)
        self.toolbox.register("individual0", tools.initRepeat, creator.Individual,
                              self.toolbox.attr0, n)
        self.toolbox.register("population0", tools.initRepeat, list, self.toolbox.individual0)

        self.toolbox.register("attr_bool", random.choices, (0, 1), k=m)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, n)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", GA2d.crossover_2point)
        self.toolbox.register("mutate", GA2d.mutation, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


if __name__ == "__main__":
    pass
