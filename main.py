import numpy as np
from activations import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    activations = [(sigmoid, de_sigmoid),
                   (tanh, de_tanh),
                   (relu, de_relu),
                   (leaky_relu, de_leaky_relu),
                   (gelu, de_gelu),
                   (swish, de_swish)]

    for (activation, de_activation) in activations:
        x, b, w = 1., -1., -1.
        y = 1.
        lr = 1e-2

        xs = np.linspace(-8, 8, 2000)
        ys = [activation(p) for p in xs]
        ds = [de_activation(p) for p in xs]

        plt.plot(xs, ys)
        plt.savefig(f'images/{activation.__name__}.png')
        plt.close()

        plt.plot(xs, ds)
        plt.savefig(f'images/{activation.__name__}_derivative.png')
        plt.close()

        x_axis = list(range(20000))
        y_axis = []
        for i in range(20000):
            z = x * w + b
            o = activation(z)
            e = (y - o) ** 2 / 2
            gw = (o - y) * de_activation(z) * x
            w -= lr * gw

            gb = (o - y) * de_activation(z)
            b -= lr * gb

            y_axis.append(o)

        plt.plot(x_axis, y_axis)
        plt.savefig(f'images/{activation.__name__}_result.png')
        plt.close()
