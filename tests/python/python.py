import pymannumopt, numpy as np

class Rosenbrock(pymannumopt.Function):
    def __init__(self, a):
        pymannumopt.Function.__init__(self)
        self.a = a

    def f(self, X):
        f = 0.
        for x, y in zip(X, X[1:]):
            u = 1-x
            v = y - x**2
            f += u**2 + self.a * v**2
        return f

    def f_fx(self, X, fx):
        f = 0.
        fx[0] = 0.
        for i, (x, y) in enumerate(zip(X, X[1:])):
            u = 1-x
            v = y - x**2
            f += u**2 + self.a * v**2
            fx[i  ] += -2*u - 4*self.a*x*v
            fx[i+1]  = 2*self.a*v
        return f

    def f_fx_fxx(self, X, fx, fxx):
        v = self.f_fx(X, fx)
        pymannumopt.numdiff.second_order_central(self, X, 1e-6, fxx)
        return v

class Quadratic(pymannumopt.Function):
    def __init__(self):
        pymannumopt.Function.__init__(self)
    def f(self, X):
        return np.sum(X**2)
    def f_fx(self, X, fx):
        fx[:] = 2*X
        return self.f(X)

class QuadraticConstraint(pymannumopt.VectorFunction):
    def __init__(self):
        pymannumopt.VectorFunction.__init__(self)
    def f(self, X, f):
        f[:] = np.sum(X**2) - 1
    def f_fx(self, X, f, fx):
        self.f(X, f)
        fx[:] = 2*X

def integrate(x, v):
    return x+v

bfgs = pymannumopt.BFGS(2)
bfgs.fxtol2 = 1e-12
bfgs.verbose = True

res, x = bfgs.minimize(Quadratic(), np.array([1., 1.]), integrate=integrate)
print(res, x)

rosenbrock = Rosenbrock(1.)
fx = np.zeros(3)
fxx = np.zeros((3,3))
print(rosenbrock.f_fx_fxx(np.ones(3), fx, fxx), fx, fxx)
res, x = bfgs.minimize(rosenbrock, np.random.random(2), integrate=integrate)
print(res, x)

al = pymannumopt.AugmentedLagrangian(2, 1)
al.etol2 = 1e-12
al.fxtol2 = 1e-10
al.maxIter = 40
al.verbose = True
bfgs.fxx_i = np.eye(2)
res, x = al.minimize(rosenbrock, QuadraticConstraint(), np.random.random(2), bfgs);
print(res, x)
