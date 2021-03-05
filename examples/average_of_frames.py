import pinocchio, pymannumopt, numpy as np
from pinocchio import liegroups
try:
    from gepetto.corbaserver import gui_client, Color
except ImportError:
    def gui_client(*args, **kwargs):
        return None

np.set_printoptions(linewidth=200)

gui = gui_client(window_name="w", dont_raise=True)

genspace = liegroups.R3() * liegroups.SO3()

Mexpected = genspace.random()
Mis = []
for _ in range(100):
    v = 0.5 * np.random.random(genspace.nv)
    Mis.append(genspace.integrate(Mexpected,  v))
    Mis.append(genspace.integrate(Mexpected, -v))

if gui:
    r = 0.002
    s = 0.002
    gui.deleteNode("M", True)
    gui.createGroup("M")
    gui.addToGroup("M", "w")
    for i, Mi in enumerate(Mis):
        name = "M/"+str(i)
        gui.addXYZaxis(name, Color.blue, r, s)
        gui.addToGroup(name, "M")
        gui.applyConfiguration(name, Mi.tolist())
    gui.addXYZaxis("expected", Color.green, r, s)
    gui.addToGroup("expected", "w")
    gui.applyConfiguration("expected", Mexpected.tolist())
    gui.refresh()

class Residual(pymannumopt.VectorFunction):
    def __init__(self, space, Mis):
        pymannumopt.VectorFunction.__init__(self)
        self.Mis = Mis
        self.space = space
        self.dim = len(self.Mis) * space.nv

    def dimension(self):
        return self.dim

    def f(self, X, f):
        for r, Mi in zip(range(0,self.dim,6), self.Mis):
            f[r:r+6] = self.space.difference(Mi, X)

    def f_fx(self, X, f, fx):
        for r, Mi in zip(range(0,self.dim,6), self.Mis):
            f[r:r+6] = self.space.difference(Mi, X)
            fx[r:r+6,:] = self.space.dDifference(Mi, X, pinocchio.ARG1)

class Error(pymannumopt.Function):
    def __init__(self, residual):
        pymannumopt.Function.__init__(self)
        self.residual = residual
        self.fd_eps = 1e-5

    def f(self, X):
        r = np.zeros((self.residual.dim))
        self.residual.f(X, r)
        return 0.5 * np.sum(r**2)

    def f_fx(self, X, fx):
        d = self.residual.dim
        r = np.zeros((d))
        rx = np.zeros((d, self.residual.space.nv))
        self.residual.f_fx(X, r, rx)
        fx[:] = np.dot(r.reshape((1, d)), rx)
        return 0.5 * np.sum(r**2)

    def f_fx_fxx(self, X, fx, fxx):
        v = self.f_fx(X, fx)
        pymannumopt.numdiff.second_order_central(self, X, self.fd_eps, fxx, integrate=self.residual.space.integrate)
        return v

for space in [ liegroups.SE3(), liegroups.R3() * liegroups.SO3(), ]:
    def check(res, x):
        assert res, "solver failed"
        if x[3] < 0 and Mexpected[3] >= 0:
            x[3:] *= -1
        np.testing.assert_almost_equal(Mexpected, x, 3)
        print(Mexpected - x)

    print("Gauss-Newton")
    residual = Residual(space, Mis)
    gn = pymannumopt.GaussNewton(7, 6)
    gn.xtol = 1e-8;
    gn.fxtol2 = 1e-11;
    gn.maxIter = 40;
    gn.verbose = True
    x0 = space.random()
    res, x = gn.minimize(residual, x0, integrate = space.integrate)
    check(res, x)

    print("BFGS")
    error = Error(residual)
    nls = pymannumopt.BFGS(7, 6)
    nls.fxtol2 = 1e-12;
    nls.maxIter = 40;
    nls.verbose = True
    res, x = nls.minimize(error, x0, integrate = space.integrate)
    check(res, x)

    print("Newton with finite diff")
    error = Error(residual)
    nls = pymannumopt.NewtonLS(7, 6)
    nls.fxtol2 = 1e-12;
    nls.maxIter = 40;
    nls.verbose = True
    res, x = nls.minimize(error, x0, integrate = space.integrate)
    check(res, x)

    if res and gui:
        name = "sol_" + space.name
        gui.addXYZaxis(name, Color.red, r, s)
        gui.addToGroup(name, "w")
        gui.applyConfiguration(name, x.tolist())
        gui.refresh()

if gui:
    gui.setCameraToBestFit("w")
