import pinocchio, pymannumopt, numpy as np

#space = pinocchio.liegroups.SE3()
space = pinocchio.liegroups.R3() * pinocchio.liegroups.SO3()

class Error(pymannumopt.VectorFunction):
    def __init__(self, Mis):
        pymannumopt.VectorFunction.__init__(self)
        self.Mis = Mis
        self.dim = len(self.Mis) * space.nv

    def dimension(self):
        return self.dim

    def f(self, X, f):
        Mx = pinocchio.XYZQUATToSE3(X)
        for r, Mi in zip(range(0,self.dim,6), self.Mis):
            f[r:r+6] = space.difference(Mi, X)

    def f_fx(self, X, f, fx):
        Mx = pinocchio.XYZQUATToSE3(X)
        for r, Mi in zip(range(0,self.dim,6), self.Mis):
            f[r:r+6] = space.difference(Mi, X)
            fx[r:r+6,:] = space.dDifference(Mi, X, pinocchio.ARG1)

Mexpected = space.random()
Mis = []
for _ in range(100):
    v = 0.1 * np.random.random(space.nv)
    Mis.append(space.integrate(Mexpected,  v))
    Mis.append(space.integrate(Mexpected, -v))

error = Error(Mis)
gn = pymannumopt.GaussNewton(7, 6)
gn.xtol = 1e-8;
gn.fxtol2 = 1e-12;
gn.maxIter = 40;
gn.verbose = True
x0 = space.random()
res, x = gn.minimize(error, x0, integrate = space.integrate)

if x[3] < 0 and Mexpected[3] >= 0:
    x[3:] *= -1
np.testing.assert_almost_equal(Mexpected, x)
