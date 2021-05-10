import math
import numpy as np
from matplotlib import pyplot as plt


_data_coef = [-2.3, 1.7, 0.8, -0.3]


def _data_fun(x):
    return (
            3 +
            _data_coef[0] * x +
            _data_coef[1] * x ** 2 +
            _data_coef[2] * x ** 3 +
            _data_coef[3] * x ** 4
    )


def sample_data(N=100, fun=_data_fun):
    rng = np.random.default_rng(1)
    x = np.sort(np.concatenate([
        rng.random(math.floor(N/5)) * 4 - 2,
        rng.random(math.floor(N/5*3)) * 4,
        rng.random(math.floor(N/5)) * 0.4 + 2
    ]))
    return (x, np.vectorize(fun)(x) + (rng.random(N) * 5 - 2.5))


def plot_data(data, plot_fun=False, fun=_data_fun):
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(data[0], data[1])
    if plot_fun:
        xlim = plt.gca().get_xlim()
        x = np.linspace(xlim[0], xlim[1], 1000)
        plt.plot(x, np.vectorize(fun)(x), c='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
