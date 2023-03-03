import math


class Node:
    """Represents a single node in computation graph."""
    def __init__(self, data, _children=(), label=''):
        self.data = data
        self.label = label
        self._children = set(_children)

    def __repr__(self):
        return f'{self.label}={self.data}'

    def __add__(self, other):
        # To support addition by a scaler, convert to node
        other = Node(other) if isinstance(other, (int, float)) else other
        out = Node(self.data + other.data, label=self.label+'+'+other.label, _children=(self, other))
        return out

    def __mul__(self, other):
        # To support multiplication by a scaler, convert to node
        other = Node(other) if isinstance(other, (int, float)) else other
        out = Node(self.data * other.data, label=self.label+'*'+other.label, _children=(self, other))
        return out


class TestNode:
    a = Node(2.0, label='a')
    b = Node(-1.5, label='b')

    def test_addition(self):
        a = self.a
        b = self.b
        assert (a + b).data == (a.data + b.data)
        assert (a + b).label == a.label+'+'+b.label
        assert (a+b)._children == {a, b}

    def test_multiplication(self):
        a = self.a
        b = self.b
        assert (a * b).data == (a.data * b.data)
        assert (a * b).label == a.label+'*'+b.label
        assert (a*b)._children == {a, b}

    def test_add_multiply(self):
        a = self.a
        b = self.b
        c = a*b
        d = c + b
        assert d.data == c.data + b.data
        assert d._children == {c, b}

    def test_node_scaler_addition(self):
        a = self.a
        assert (a + 1.0).data == (a.data + 1.0)

    def test_node_scalar_multiplication(self):
        a = self.a
        assert (a*1.5).data == (a.data*1.5)


if __name__ == '__main__':
    pass
