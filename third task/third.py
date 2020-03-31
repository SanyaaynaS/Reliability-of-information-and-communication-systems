from matplotlib import pyplot as plt
import math
import numpy as np
import time


def generate_time(n: np.int, p: np.ndarray, l: np.ndarray, seed_init: np.int):
    np.random.seed(seed_init)
    for sys in range(n):
        u = np.random.random_sample()
        i, s = 0, p[0]
        while u > s:
            i += 1
            s += p[i]
        t_arr = np.array([-np.log(np.random.random_sample()) / l[i] for i in range(len(l))])
        yield t_arr[i], np.min(t_arr), np.max(t_arr)


def simulation(args: tuple, time: np.ndarray, delta):
    n_t = np.zeros((len(time), 3), dtype=np.float64)
    for ind, cur_time in enumerate(time):
        for g_t1, g_t2, g_t3 in generate_time(*args):
            if cur_time <= g_t1:
                n_t[ind][0] += 1

            if cur_time <= g_t2:
                n_t[ind][1] += 1

            if cur_time <= g_t3:
                n_t[ind][2] += 1

    l_t = np.zeros((len(time), 3), dtype=np.float64)
    for j in range(3):
        l_t[:, j] = -np.gradient(n_t[:, j], delta, edge_order=2) / (n_t[:, j] + 1e-4)

    return n_t / args[0], l_t


def make_exp_val(p, l, time, delta):
    exp = np.full((len(p),), math.exp(1))
    r1 = np.zeros((len(time),))
    r2 = np.zeros((len(time),))

    for i, t in enumerate(time):
        r1[i] = (exp ** (-l * t) * p).sum()
        r2[i] = np.e ** (-l.sum() * t)
    l1 = -np.gradient(r1, delta, edge_order=2) / r1
    l2 = np.full((len(time),), l.sum())

    r3 = np.e ** (-l[0] * time) + np.e ** (-l[1] * time) - np.e ** (-l[0] * time) * np.e ** (-l[1] * time)
    l3 = -np.gradient(r3, delta, edge_order=2) / r3

    return (r1, l1), (r2, l2), (r3, l3)


if "__main__" == __name__:
    l1, p1 = 0.7, 0.3
    l2, p2 = 0.9, 0.7
    t = np.array([i / 10 for i in range(50)], dtype=float)

    num_systems = 10 ** 4
    probs = np.array([p1, p2], dtype=np.float64)
    lambdas = np.array([l1, l2], dtype=np.float64)
    s_init = np.int(np.round(time.time()))
    d = 0.1

    s1, s2, s3 = make_exp_val(np.array([p1, p2]), np.array([l1, l2]), t, d)

    R, lam = simulation((num_systems, probs, lambdas, s_init), t, d)

    n_fig = 1

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s1[0], 'o-')
    plt.plot(t, R[:, 0], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$R(t)$')
    plt.title('Период приработки, функция надежности')
    plt.show()

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s1[1], 'o-')
    plt.plot(t, lam[:, 0], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$\lambda(t)$')
    plt.title('Период приработки, функция интенсивности отказов')
    plt.grid()
    plt.show()

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s2[0], 'o-')
    plt.plot(t, R[:, 1], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$R(t)$')
    plt.title('Период н.ф., функция надежности')
    plt.grid()
    plt.show()

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s2[1], 'o-')
    plt.plot(t, lam[:, 1], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$\lambda(t)$')
    plt.title('Период н.ф., функция интенсивности отказов')
    plt.grid()
    plt.show()

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s3[0], 'o-')
    plt.plot(t, R[:, 2], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$R(t)$')
    plt.title('Период старения, функция надежности')
    plt.grid()
    plt.show()

    plt.figure(n_fig)
    n_fig += 1
    plt.plot(t, s3[1], 'o-')
    plt.plot(t, lam[:, 2], 'o-')
    plt.legend(('теория', 'практика'))
    plt.xlabel('t')
    plt.ylabel('$\lambda(t)$')
    plt.title('Период старения, функция интенсивности отказов')
    plt.grid()
    plt.show()
