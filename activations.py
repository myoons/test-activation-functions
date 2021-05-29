import math


def sigmoid(z):
    return 1 / (1 + math.e ** (-z))


def de_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return (1 - math.e ** (-z)) / (1 + math.e ** (-z))


def de_tanh(z):
    return (1 - tanh(z)) * (1 + tanh(z))


def relu(z):
    return max(z, 0)


def de_relu(z):
    if z < 0:
        return 0
    else:
        return 1


def leaky_relu(z):
    return max(z, 0.01 * z)


def de_leaky_relu(z):
    if z < 0:
        return 0.01
    else:
        return 1


def gelu(z):
    cdf = 0.5 * (1.0 + tanh(math.sqrt(2 / math.pi) * (z + 0.004715 * math.pow(z, 3))))
    return z * cdf


def de_gelu(z):
    cdf = 0.5 * (1.0 + tanh(math.sqrt(2 / math.pi) * (z + 0.004715 * math.pow(z, 3))))
    return cdf + z * 0.5 * de_tanh(math.sqrt(2 / math.pi) * (z + 0.004715 * math.pow(z, 3))) * math.sqrt(2 / math.pi) * (1 + 3 * 0.004715 * math.pow(z, 2))


def swish(z):
    return z * sigmoid(z)


def de_swish(z):
    return sigmoid(z) + z * de_sigmoid(z)