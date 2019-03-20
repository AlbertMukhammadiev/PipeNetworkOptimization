"""
- GA(100 поколений, 100 особей)
- Шестиугольная сетка
- Двумерная нумерация по построению, двумерные эволюционные операторы
"""

from network import NetworkGA2D, Point
from layouts import HexagonGrid
from ga import GA2d


layout = HexagonGrid(10, 10, scale=0.2, start=Point(0, -1))
cost_model = [dict(diam=0.0,   cost=0.0),
              dict(diam=80.0,  cost=23.0),
              dict(diam=100.0, cost=32.0),
              dict(diam=120.0, cost=50.0),
              dict(diam=140.0, cost=60.0),
              dict(diam=160.0, cost=90.0),
              dict(diam=180.0, cost=130.0),
              dict(diam=200.0, cost=170.0),
              dict(diam=220.0, cost=300.0),
              dict(diam=240.0, cost=340.0),
              dict(diam=260.0, cost=390.0),
              dict(diam=280.0, cost=430.0),
              dict(diam=300.0, cost=470.0),
              dict(diam=320.0, cost=500.0)]


network = NetworkGA2D(layout, cost_model)
network.add_sink(demand=-120, pos=Point(x=2, y=2))
network.add_source(demand=10, pos=Point(x=0, y=0))
network.add_source(demand=20, pos=Point(x=1, y=0))
network.add_source(demand=10, pos=Point(x=0, y=1))
network.add_source(demand=20, pos=Point(x=2, y=0))
network.add_source(demand=10, pos=Point(x=1, y=1))
network.add_source(demand=20, pos=Point(x=0, y=2))
network.add_source(demand=10, pos=Point(x=2, y=1))
network.add_source(demand=20, pos=Point(x=1, y=2))

algorithm = GA2d(network)
algorithm.n_individuals = 50
algorithm.n_generations = 10
algorithm.run()
