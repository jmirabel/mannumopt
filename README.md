## mannumopt

It is a *num*erical *opt*imization on *man*ifolds. It provides tools to solve
- equality constrained non-linear minimization problems,
- equality constrained least square problems.

It implements :
- BFGS
- Newton
- Gauss-Newton
- Augmented Lagrangian
- Squared penalty

For some examples, see the unit test or the example folder.

Author: Joseph Mirabel

## Disclaimer

This project was initially started as a practical exercise while reading the excellent [numerical optimization] book. It targets small dimensional dense problems, on manifolds.

- If your problem is on vector space, you should probably use a more mature library.
- If your problem is on manifolds and deals with large sparse matrices, you should use another library, such as [Google Ceres].
- As of writing, the trust-region algorithms are not mature.

[numerical optimization]: <http://users.iems.northwestern.edu/~nocedal/book/num-opt.html> "Numerical Optimization, by Jorge Nocedal and Stephen J. Wright"
[Google Ceres]: http://ceres-solver.org/
