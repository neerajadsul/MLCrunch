from engine import Node
import math

from pytest import approx


class TestNode:

    def test_addition(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        assert (a + b).data == (a.data + b.data)
        assert (a + b).label == a.label+'+'+b.label
        assert (a+b)._children == {a, b}

    def test_multiplication(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        assert (a * b).data == (a.data * b.data)
        assert (a * b).label == a.label+'*'+b.label
        assert (a*b)._children == {a, b}

    def test_add_multiply(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        c = a*b
        d = c + b
        assert d.data == c.data + b.data
        assert d._children == {c, b}

    def test_node_scaler_addition(self):
        a = Node(2.0, label='a')
        assert (a + 1.0).data == (a.data + 1.0)

    def test_node_scalar_multiplication(self):
        a = Node(2.0, label='a')
        assert (a*1.5).data == (a.data*1.5)

    def test_tanh(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        assert a.tanh().data == math.tanh(a.data)
        assert b.tanh().data == math.tanh(b.data)
        assert a.tanh()._children == {a}

    def test_add_backprop(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        c = a + b
        c.grad = 1
        c._backprop()
        assert a.grad == 1 and b.grad == 1, f'{a.grad},{b.grad},{c.grad}'

    def test_mult_backprop(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        c = a * b
        c.backprop()
        assert a.grad == b.data
        assert b.grad == a.data

    def test_tanh_backprop(self):
        a = Node(2.0, label='a')
        c = a.tanh()
        c.grad = 1
        c.backprop()
        assert a.grad == (1-c.data**2), (1-c.data**2)

    def test_full_backprop(self):
        x1 = Node(2.0, label='x1')
        x2 = Node(0.0, label='x2')
        w1 = Node(-3.0, label='w1')
        w2 = Node(1.0, label='w2')

        b = Node(6.8813735881, label='b')

        # Forward propagation
        a1 = w1 * x1
        a2 = w2 * x2
        c = a1 + a2
        n = c + b
        o = n.tanh()
        # Backpropagation
        o.backprop()

        assert w1.grad == approx(1.0)
        assert x1.grad == approx(-1.5)
        assert w2.grad == approx(0.0)
        assert x2.grad == approx(0.5)

    def test_mult_same_node(self):
        a = Node(2.0, label='a')
        c = a * a
        c.backprop()
        assert a.grad == 2*a.data

    def test_add_same_node(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        c = a + a + b
        c.backprop()
        assert a.grad == 2

    def test_exp_backprop(self):
        a = Node(2.0, label='a')
        c = a.exp()
        c.backprop()
        assert a.grad == math.exp(a.data)

    def test_pow_positive_backprop(self):
        a = Node(2.0, label='a')
        c = a**2
        c.backprop()
        assert c.data == approx(a.data**2)
        assert a.grad == approx(2*a.data)

    def test_pow_negative_backprop(self):
        a = Node(2.0, label='a')
        c = a**-2
        c.backprop()
        assert c.data == approx(1/a.data**2)
        assert a.grad == approx(-2*a.data**-3)

    def test_truediv_backprop(self):
        a = Node(2.0, label='a')
        b = Node(-1.5, label='b')
        c = a/b
        c.backprop()
        assert a.grad == (1/b.data)
        assert b.grad == approx(-1*a.data/b.data**2)

    def test_neg_backprop(self):
        a = Node(2.0, label='a')
        c = -a
        c.backprop()
        assert c.data == -a.data
        assert a.grad == -1


