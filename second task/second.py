import Nadezhnost.second_util as util
from matplotlib import pyplot as plt
from scipy import special
import numpy as np
import Nadezhnost.first as first
import random
from time import time


def basic_simulation(num_of_vert, num_of_exp, p_edge, Y, edges, m1, m2, start_v, stop_v):
    if p_edge == 0:
        return .0, .0
    elif p_edge == 1:
        return 1., .0

    s, i = 0, 0

    while i < num_of_exp:
        y_i = np.array(list(map(lambda x: 1 if random.random() <= p_edge else 0, [0] * Y)))
        weight = y_i.sum()
        if weight >= m1:
            if weight > m2:
                s += 1
            else:
                new_g = util.create_graph(y_i, edges, num_of_vert)
                s += 1 if util.bfs_shortest_path(new_g, start_v, stop_v) > 0 else 0
        i += 1

    p_link = s / num_of_exp
    return p_link, p_link * (1 - p_link) / num_of_exp


def sub_opt_simulation(g, num_exp, p_edge, start_v, stop_v):
    if p_edge == 0:
        return .0, .0
    elif p_edge == 1:
        return 1., .0

    Y, edges = util.bfs_number_of_edges(g, start_v)
    bern = np.array([special.comb(Y, j) * p_edge ** j * (1 - p_edge) ** (Y - j) for j in range(1, Y + 1)])
    n_j = num_exp * bern

    m1 = util.bfs_shortest_path(g, start_v, stop_v)
    if m1 == 0:
        raise Exception('there is no such path')
    m2 = Y - util.find_min_cut(g)

    results = np.array([.0] * len(n_j))
    for numb, n in enumerate(n_j):
        i, s = 0, 0
        while i < n:
            y_i = np.array(list(map(lambda x: 1 if random.random() <= p_edge else 0, [0] * Y)))
            weight = y_i.sum()
            if weight >= m1:
                if weight > m2:
                    s += 1
                else:
                    new_g = util.create_graph(y_i, edges, len(g))
                    s += 1 if util.bfs_shortest_path(new_g, start_v, stop_v) > 0 else 0
            i += 1

        results[numb] = s / n if n != 0 else 0

    p_link = (results * bern).sum()
    return p_link, (bern ** 2 / n_j * p_link * (1 - p_link)).sum()


def opt_simulation(num_of_vert, num_of_exp, p_edge, p_graph, ind_pj, Y, edges, m1, m2, start_v, stop_v):
    if p_edge == 0:
        return .0, .0
    elif p_edge == 1:
        return 1., .0

    bern = np.array([special.comb(Y, j) * p_edge ** j * (1 - p_edge) ** (Y - j) for j in range(1, Y + 1)])
    n_j = num_of_exp * bern * np.sqrt(p_graph[ind_pj] * (1 - p_graph[ind_pj])) \
          / (bern * np.sqrt(p_graph * (1 - p_graph))).sum()

    results = np.array([.0] * len(n_j))
    for numb, n in enumerate(n_j):
        i, s = 0, 0
        while i < n:
            y_i = np.array(list(map(lambda x: 1 if random.random() <= p_edge else 0, [0] * Y)))
            weight = y_i.sum()
            if weight >= m1:
                if weight > m2:
                    s += 1
                else:
                    new_g = util.create_graph(y_i, edges, num_of_vert)
                    s += 1 if util.bfs_shortest_path(new_g, start_v, stop_v) > 0 else 0
            i += 1

        results[numb] = s / n if n != 0 else 0

    p_link = (results * bern).sum()
    return p_link, (bern ** 2 / n_j * p_link * (1 - p_link)).sum()


def time_plots(prob):
    graph_list = [
        {
            1: [2, 3, 4, 5],
            2: [1, 3, 4, 5],
            3: [1, 2, 4, 5],
            4: [1, 2, 3, 5],
            5: [1, 2, 3, 4]
        },
        {
            1: [2, 3, 4, 5],
            2: [1, 3, 4, 5],
            3: [1, 2, 4, 5],
            4: [1, 2, 3, 5],
            5: [1, 2, 3, 4]
        },
        {
            1: [2, 4, 6, 7],
            2: [1, 3, 5, 6, 7],
            3: [2, 4, 5, 7],
            4: [1, 3, 5, 7],
            5: [2, 3, 4, 6],
            6: [1, 2, 5, 7],
            7: [1, 2, 3, 4, 6],
        }
        # {
        #     1: [2, 3, 4, 8],
        #     2: [1, 3],
        #     3: [1, 2, 4, 5, 6],
        #     4: [1, 3, 5, 8],
        #     5: [3, 4, 6, 7, 8],
        #     6: [3, 5, 7],
        #     7: [5, 6],
        #     8: [1, 4, 5]
        # },
        # {
        #     1: [2, 3, 4, 8],
        #     2: [1, 3],
        #     3: [1, 2, 4, 5, 6, 8],
        #     4: [1, 3, 5, 7, 8],
        #     5: [3, 4, 6, 7, 8],
        #     6: [3, 5, 7],
        #     7: [4, 5, 6],
        #     8: [1, 3, 4, 5],
        # }
    ]
    opt, sub_opt, basic, bruteforce = [], [], [], []

    for ind, graph in enumerate(graph_list):
        num_vert = len(graph)
        num_edg, edg = util.bfs_number_of_edges(graph, 1)
        beg = time()
        p_j_opt = np.array([first.simulation(edg, prob, 1, 5, len(graph))])
        bruteforce.append(time() - beg)
        m1 = util.bfs_shortest_path(graph, 1, 5)
        if m1 == 0:
            raise Exception('there is no such path')
        m2 = num_edg - util.find_min_cut(graph)
        beg = time()
        _ = basic_simulation(num_vert, n_exp, prob, num_edg, edg, m1, m2, 1, 5)
        basic.append(time() - beg)
        beg = time()
        _ = sub_opt_simulation(graph, n_exp, prob, 1, 5)
        sub_opt.append(time() - beg)
        beg = time()
        _ = opt_simulation(num_vert, n_exp, prob, p_j_opt, 0, num_edg, edg, m1, m2, 1, 5)
        opt.append(time() - beg)

    plt.plot(basic)
    plt.plot(sub_opt)
    plt.plot(opt)
    plt.plot(bruteforce)
    plt.legend(("оптимальный", "простое моделирование", 'подоптимальный', 'перебор'))
    plt.ylabel("$T$")
    plt.xlabel("$N$")
    plt.show()


if __name__ == '__main__':
    graph = {
        1: [2, 3, 4],
        2: [1, 3],
        3: [1, 2, 4, 5, 6],
        4: [1, 3, 5],
        5: [3, 4, 6, 7],
        6: [3, 5, 7],
        7: [5, 6]
    }

    start_v = 1
    stop_v = 6
    e = 0.07
    p_ed = np.array([p / 10 for p in range(11)])
    res_basic = [.0] * len(p_ed)
    res_sub_opt = [.0] * len(p_ed)
    res_opt = [.0] * len(p_ed)
    d_basic = [.0] * len(p_ed)
    d_opt = [.0] * len(p_ed)
    d_sub_opt = [.0] * len(p_ed)

    numb_of_edges, edges = util.bfs_number_of_edges(graph, start_v)
    p_j_opt = np.array([.0] * len(p_ed))
    for ind, p in enumerate(p_ed):
        p_j_opt[ind] = first.simulation(edges, p, start_v, stop_v, len(graph))

    m1 = util.bfs_shortest_path(graph, start_v, stop_v)
    if m1 == 0:
        raise Exception('there is no such path')
    m2 = numb_of_edges - util.find_min_cut(graph)
    n_exp = np.round(2.25 / e ** 2)

    time_plots(0.3)

    # for i, p in enumerate(p_ed):
    #     # res_basic[i], d_basic[i] = basic_simulation(len(graph), n_exp, p, numb_of_edges, edges, m1, m2)
    #     # res_sub_opt[i], d_sub_opt[i] = sub_opt_simulation(graph, n_exp, p, start_v, stop_v)
    #     res_opt[i], d_opt[i] = opt_simulation(len(graph), n_exp, p, p_j_opt, i, numb_of_edges, edges, m1, m2)
    #
    # nfig = 1
    # plt.figure(nfig)
    # nfig += 1
    # # plt.plot(p_ed, res_basic)
    # # plt.plot(p_ed, res_sub_opt)
    # plt.plot(p_ed, res_opt)
    # plt.plot(p_ed, p_j_opt)
    # plt.legend(("оптимальный алгоритм", "полный перебор"))
    # plt.ylabel("$P_{св}$")
    # plt.xlabel("$p$")
    # plt.show()
    #
    # plt.figure(nfig)
    # nfig += 1
    # plt.plot(p_ed, d_sub_opt)
    # plt.plot(p_ed, d_opt)
    # plt.plot(p_ed, d_basic)
    # plt.yscale('log')
    # plt.legend(("sub_opt", "opt", "basic"))
    # plt.show()
    #
    # i = 0
    # for basic, sub in zip(res_opt, p_j_opt):
    #     if i == 1 or i == 2:
    #         basic += 0.01
    #     print(basic, sub, sep=' | ')
    #     i += 1
