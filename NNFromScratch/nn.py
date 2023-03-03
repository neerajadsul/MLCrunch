import random
from engine import Node


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Node(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w)
        out = self.b
        for wi, xi in zip(self.w, x):
            out += wi * xi
        return out.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        for p in self.neurons:
            p.zero_grad()


class MLP:
    def __init__(self, n_in=3, n_hidden=[3, 3], n_out=1):
        # layer sizes list
        sz = [n_in] + n_hidden + [n_out]
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.layers:
            p.zero_grad()


if __name__ == '__main__':
    x = [2.0, 3, -1]
    n = MLP(len(x), [4, 4], 1)
    print(n(x))
