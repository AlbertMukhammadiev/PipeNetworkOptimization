import random

from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

from network import Network

log = {
    'min': [],
    'max': [],
    'avg': [],
}

class Runner:
    def __init__(self, network: Network) -> None:
        self._individual_size = network.nbits_required()
        self._network = network
        
        self._n_generations = 100
        self._n_individuals = 100
        self.configure_algorithm()

    n_generations = property()
    n_individuals = property()

    @n_generations.setter
    def n_generations(self, n):
        if n < 100:
            self._n_generations = 100
        elif n > 1000:
            self._n_generations = 1000
        else:
            self._n_generations = n

    @n_individuals.setter
    def n_individuals(self, n):
        if n < 30:
            self._n_individuals = 30
        elif n > 200:
            self._n_individuals = 200
        else:
            self._n_individuals = n

    def configure_algorithm(self):
        def evalOneMax(individual):
            return self._network.total_cost(individual),

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

    def save_best(self, population, generation_No):
        best_ind = tools.selBest(population, 1)[0]
        self._network.redesign(best_ind)
        self._network.draw(f'best_in_{generation_No}_generation')
    
    def save_worst(self, population, generation_No):
        worst_ind = tools.selWorst(population, 1)[0]
        self._network.redesign(worst_ind)
        self._network.draw(f'worst_in_{generation_No}_generation')
        
    def main(self):
        random.seed(32)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=self._n_individuals)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.1
        
        print("Start of evolution")
        
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of 
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        
        # Begin the evolution
        # while max(fits) < 100 and g < self._ngenerations:
        while g < self._n_generations:
            self.save_best(pop, g)
            self.save_worst(pop, g)
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)
            
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            # print("  Std %s" % std)
            log['min'].append(min(fits))
            log['max'].append(max(fits))
            log['avg'].append(mean)

        
        print("-- End of (successful) evolution --")
        
        best_ind = tools.selBest(pop, 1)[0]
        self._network.redesign(best_ind)
        self._network.draw('best')

        print(best_ind.fitness.values[0])
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    from data_context import DataContext
    from matplotlib.pyplot import scatter, show, Figure

    path = 'projects/square_layout/'
    dcontext = DataContext(path)
    network = Network(dcontext)
    network.draw('initial')
    runner = Runner(network)
    runner.main()
    
    xs = range(len(log['min']))
    
    fig = Figure()
    scatter(xs, log['min'])
    plt.savefig('minimum.png', format='PNG')
    plt.close()

    fig = Figure()
    scatter(xs, log['max'])
    plt.savefig('maximum.png', format='PNG')
    plt.close()

    fig = Figure()
    scatter(xs, log['avg'])
    plt.savefig('avg.png', format='PNG')
    plt.close()