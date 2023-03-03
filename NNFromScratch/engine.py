import math


class Node:
    """Represents a single node in computation graph."""
    def __init__(self, data, _children=(), label=''):
        self.data = data
        self.label = label
        self._children = set(_children)

        self.grad = 0.0
        self._backprop = lambda: None

    def __repr__(self):
        return f'{self.label}={self.data}'

    def __add__(self, other):
        # To support addition by a scaler, convert to node
        other = Node(other) if isinstance(other, (int, float)) else other
        out = Node(self.data + other.data, label=self.label+'+'+other.label, _children=(self, other))

        def _backprop():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backprop = _backprop
        return out

    def __mul__(self, other):
        # To support multiplication by a scaler, convert to node
        other = Node(other) if isinstance(other, (int, float)) else other
        out = Node(self.data * other.data, label=self.label+'*'+other.label, _children=(self, other))

        def _backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backprop = _backprop
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Node(t, _children=(self,), label=f'tanh({self.label})')

        def _backprop():
            self.grad += (1 - t**2) * out.grad
        out._backprop = _backprop
        return out

    def backprop(self):
        """Back-propagate through all child nodes from front to back"""
        computation_graph = []
        visited = set()

        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_graph(child)
                computation_graph.append(v)

        build_graph(self)
        self.grad = 1.0
        for node in reversed(computation_graph):
            node._backprop()


if __name__ == '__main__':
    a = Node(2.0)
    b = Node(-1.5)
    c = a * b
    c.backprop()
    print(a.grad, b.grad)
