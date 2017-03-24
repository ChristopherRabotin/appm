import cvxpy as cvx
import numpy as np
from scipy.optimize import fmin

from gradient import descend

def fNg(curX):
    fx = 0.5*(np.linalg.norm(A.dot(curX)-b, 2))**2
    gx = (A.transpose().dot(A)).dot(curX) - A.transpose().dot(b)
    return fx, gx

# The following function is copied from the solutions (not a package, can't import)
def project_l1(x, tau=1.):
    """
    project_l1(x, tau) -> y
      projects x onto the scaled l1 ball, ||x||_1 <= tau
      If tau is not provided, the default is tau = 1.

    Stephen Becker and Emmanuel Candes, 2009/2010.
    Crucial bug fix: 3/17/2017, SRB
    """
    absx = np.abs(x)
    s = np.sort(absx)[::-1] # sort in descending order
    cs = np.cumsum(s)

    if cs[-1] <= tau:
        # x is already feasible, so no thresholding needed
        return x

    # Check some "discrete" levels of shrinkage, e.g. [s(2:end),0]
    # This lets us discover which indices will be nonzero
    n = x.size
    i_tau = np.where(cs -
        np.arange(1,n+1)*np.concatenate((s[1:],0), axis=None) >= tau)[0][0]

    # Now that we know which indices are involved, it's a very simple problem
    thresh = (cs[i_tau]-tau) / (i_tau+1)

    # Shrink x by the amount "thresh"
    return np.sign(x)*np.maximum(absx - thresh, 0)

if __name__ == '__main__':

    A = np.random.rand(10, 20)
    b = np.random.rand(10,1)
    x = cvx.Variable(20)

    # Solving BP
    bp = cvx.Problem(cvx.Minimize(cvx.norm1(x)), [A*x == b])
    bp.solve(verbose=False)
    #print(bp.status, x.value)

    # Solving LSt
    tau = np.linalg.norm(x.value, 1)
    print('tau = {}'.format(tau))
    x0 = np.random.rand(20, 1)

    x_values, f0_values, g0_values = descend(fNg, x0, tol=1e-3, tolX=1e-6, linesearch=True, heavy=True, prox=lambda x, t, g: project_l1(x, tau))
    print(x_values[-1])
    print(A.dot(x_values[-1])+b)
    print(np.linalg.norm(x_values[-1] - x.value, 2))
