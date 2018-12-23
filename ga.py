#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1
import random

from deap import base
from deap import creator
from deap import tools

from network import Network
from reader import ReaderCSV, FileNames

log = {
    'min': [],
    'max': [],
    'avg': [],
}

class Runner:
    def __init__(self, network: Network) -> None:
        self._individual_size = network.nbits_required()
        self._network = network
        
        self._ngenerations = 50
        self._nindividuals = 100
        self.configure_algorithm()

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

    def main(self):
        random.seed(20)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=self._nindividuals)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.2
        
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
        while g < self._ngenerations:
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
            
            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            log['min'].append(min(fits))
            log['max'].append(max(fits))
            log['avg'].append(mean)

        
        print("-- End of (successful) evolution --")
        
        best_ind = tools.selBest(pop, 1)[0]
        # self._network.draw('best')
        print(best_ind.fitness.values[0])
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    
    FNAME_NODES = 'initial_layouts/square/nodes.csv'
    FNAME_EDGES = 'initial_layouts/square/edges.csv'
    FNAME_COSTS = 'initial_layouts/square/cost_data.csv'
    fnames = FileNames(
        nodes=FNAME_NODES,
        edges=FNAME_EDGES,
        costs=FNAME_COSTS,
    )

    reader = ReaderCSV(fnames)
    network = Network(reader)
    runner = Runner(network)
    runner.main()
    from matplotlib.pyplot import scatter, show
    xs = range(len(log['min']))
    scatter(xs, log['max'])
    show()
    