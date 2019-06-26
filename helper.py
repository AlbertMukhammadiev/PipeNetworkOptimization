import math
import random
from itertools import product
import networkx as nx
import numpy as np
from functools import partial


round2 = partial(round, ndigits=2)


def distance(p1, p2):
    return math.hypot(p1[1] - p2[1], p1[0] - p2[0])


def random_point(a, b):
    x = random.randint(a, b)
    y = random.randint(a, b)
    return x, y


def binaries(n):
    k = math.ceil(math.log(n, 2))
    keys = list(range(2 ** k))
    return list(map(lambda key: bin(key)[2:].zfill(k), keys))


def binaries_Gray(n):
    return binaries(n)


def brute_points(points):
    if points:
        xmax = (max(points, key=lambda p: p[0]))[0]
        xmin = (min(points, key=lambda p: p[0]))[0]
        ymax = (max(points, key=lambda p: p[1]))[1]
        ymin = (min(points, key=lambda p: p[1]))[1]
        dx = (xmax - xmin) / 50
        dy = (ymax - ymin) / 50
        xs = np.arange(xmin, xmax, dx)
        ys = np.arange(ymin, ymax, dy)
        return product(xs, ys)
    else:
        return []
