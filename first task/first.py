import itertools
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import Nadezhnost.second_util as util


def check_the_result(p, poly):
    poly_prob = [p ** i for i in range(len(poly))]
    res = 0
    for prob, c in zip(poly_prob, reversed(poly)):
        res += prob * c

    return res


def make_poly():
    p = symbols('p')
    x = p + p ** 2 - p ** 3
    px = p + x - p * x
    pp = 2 * p - p ** 2

    poly_a = Poly(px * (p + pp * x - p * pp * x), p)
    poly_b = Poly((x + p ** 2 - x * p ** 2) * px, p)
    poly_c = Poly((x * p + p ** 2 * x - p ** 3 * x ** 2), p)

    return Poly(poly_a * p + (poly_b * p + poly_c * (1 - p)) * (1 - p), p)


def simulation(pairs, p_edge, start, stop, num_of_vertexes):
    comb_vectors = itertools.product([0, 1], repeat=len(pairs))
    p_total = 0

    for i, case in enumerate(comb_vectors):
        coord = np.argwhere(np.array(case) == 1).ravel()
        existing_edges = coord.shape[0]

        g = util.create_graph(case, pairs, num_of_vertexes)
        if util.bfs_shortest_path(g, start, stop) != 0:
            p_total += p_edge ** existing_edges * (1 - p_edge) ** (len(pairs) - existing_edges)

    return p_total


if __name__ == '__main__':
    pairs = [[1, 2], [1, 3], [1, 4], [2, 3], [3, 4], [3, 6], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7]]
    num_of_vert = 7
    start_v, stop_v = 1, 6

    prob_of_edges = np.linspace(0, 1, 101)
    y = np.zeros(len(prob_of_edges))
    check_arr = np.zeros(len(prob_of_edges))
    poly = make_poly()
    for ind, p in enumerate(prob_of_edges):
        y[ind] = simulation(pairs, p, start_v, stop_v, num_of_vert)
        check_arr[ind] = check_the_result(p, poly.all_coeffs())

    plt.plot(prob_of_edges, y, prob_of_edges, check_arr)
    plt.legend(("перебор", "декомпозиция"))
    plt.xlabel("$p$")
    plt.ylabel("$P_{CB}$")
    plt.show()

    for sim, th in zip(y, check_arr):
        print(sim, th, sep=' | ')
