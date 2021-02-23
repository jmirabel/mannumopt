import unittest
import pymannumopt, numpy as np

class QuadraticFD(pymannumopt.Function):
    def __init__(self):
        pymannumopt.Function.__init__(self)
    def f(self, X):
        return np.sum(X**2)
    def f_fx(self, X, fx):
        pymannumopt.numdiff.central(self, X, 1e-5, fx)
        return self.f(X)
    def f_fx_fxx(self, X, fx, fxx):
        v = self.f(X)
        pymannumopt.numdiff.forward(self, X, 1e-5, fx)
        pymannumopt.numdiff.second_order_central(self, X, 1e-5, fxx)
        return v

class TestFunction(unittest.TestCase):
    def test_value_consistency(self):
        quad = QuadraticFD()
        X = np.random.random(3)
        fx = np.zeros(3)
        fxx = np.zeros((3,3))
        v = quad.f(X)

        self.assertEqual(v, quad.f_fx(X, fx))
        np.testing.assert_almost_equal(fx, 2*X)

        self.assertEqual(v, quad.f_fx_fxx(X, fx, fxx))
        np.testing.assert_almost_equal(fx, 2*X, 5)
        np.testing.assert_almost_equal(fxx, 2*np.eye(3), 5)

if __name__ == '__main__':
    unittest.main()
