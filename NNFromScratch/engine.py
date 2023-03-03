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
        out = Node(self.data + other.data, label=self.label+'+'+other.label, _children=(self, other))
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


if __name__ == '__main__':
    pass
