"""
- GA(100 поколений, 100 особей)
- Квадратная сетка 100 узлов
- Одномерная нумерация по построению
"""

from network import NetworkGA, Point
from layouts import SquareGrid
from ga import GA


layout = SquareGrid(11, 11, 0.2)
cost_model = [dict(diameter=0.0, cost=0.0),
              dict(diameter=80.0, cost=23.0),
              dict(diameter=100.0, cost=32.0),
              dict(diameter=120.0, cost=50.0),
              dict(diameter=140.0, cost=60.0),
              dict(diameter=160.0, cost=90.0),
              dict(diameter=180.0, cost=130.0),
              dict(diameter=200.0, cost=170.0),
              dict(diameter=220.0, cost=300.0),
              dict(diameter=240.0, cost=340.0),
              dict(diameter=260.0, cost=390.0),
              dict(diameter=280.0, cost=430.0),
              dict(diameter=300.0, cost=470.0),
              dict(diameter=320.0, cost=500.0)]

network = NetworkGA(layout, cost_model)
network.add_sink(demand=-120, position=Point(x=2, y=2))
network.add_source(demand=10, position=Point(x=0, y=0))
network.add_source(demand=20, position=Point(x=1, y=0))
network.add_source(demand=10, position=Point(x=0, y=1))
network.add_source(demand=20, position=Point(x=2, y=0))
network.add_source(demand=10, position=Point(x=1, y=1))
network.add_source(demand=20, position=Point(x=0, y=2))
network.add_source(demand=10, position=Point(x=2, y=1))
network.add_source(demand=20, position=Point(x=1, y=2))

algorithm = GA(network)
algorithm.n_individuals = 100
algorithm.n_generations = 100
algorithm.run()
