"""
- GA(100 поколений, 100 особей)
- Квадратный макет, все узлы продуктивные
- Одномерная нумерация по построению
- Все узлы имеют потоки
"""

from network import NetworkGA, Point
from layouts import Layout
from ga import GA

layout = Layout()
layout.add_node(node=1, position=Point(x=0, y=0))
layout.add_node(node=2, position=Point(x=1, y=0))
layout.add_node(node=3, position=Point(x=0, y=1))
layout.add_node(node=4, position=Point(x=2, y=0))
layout.add_node(node=5, position=Point(x=1, y=1))
layout.add_node(node=6, position=Point(x=0, y=2))
layout.add_node(node=7, position=Point(x=2, y=1))
layout.add_node(node=8, position=Point(x=1, y=2))
layout.add_node(node=9, position=Point(x=2, y=2))

layout.add_edge(1, 2)
layout.add_edge(1, 3)
layout.add_edge(2, 4)
layout.add_edge(2, 5)
layout.add_edge(3, 5)
layout.add_edge(3, 6)
layout.add_edge(4, 7)
layout.add_edge(5, 7)
layout.add_edge(5, 8)
layout.add_edge(6, 8)
layout.add_edge(7, 9)
layout.add_edge(8, 9)

layout.draw_in_detail()

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

network = NetworkGA(layout, cost_model)

network.add_sink(demand=-120, pos=Point(x=2, y=2))
network.add_source(demand=10, pos=Point(x=0, y=0))
network.add_source(demand=20, pos=Point(x=1, y=0))
network.add_source(demand=10, pos=Point(x=0, y=1))
network.add_source(demand=20, pos=Point(x=2, y=0))
network.add_source(demand=10, pos=Point(x=1, y=1))
network.add_source(demand=20, pos=Point(x=0, y=2))
network.add_source(demand=10, pos=Point(x=2, y=1))
network.add_source(demand=20, pos=Point(x=1, y=2))

algorithm = GA(network)
algorithm.n_individuals = 300
algorithm.n_generations = 100
algorithm.run()
