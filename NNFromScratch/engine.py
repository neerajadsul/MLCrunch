import math


class Node:
    """Represents a single node in computation graph."""
    def __init__(self, data, label=''):
        self.data = data
        self.label = label

    def __repr__(self):
        return f'{self.label}={self.data}'

    