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
    if type(x[0]) == np.ndarray:
        x = x[0] # TODO: Fix for matrices
    absx = np.abs(x)
    s = np.sort(absx)[::-1] # sort in descending order
    cs = np.cumsum(s)

    if cs[-1] <= tau:
        # x is already feasible, so no thresholding needed
        return x

    # Check some "discrete" levels of shrinkage, e.g. [s(2:end),0]
    # This lets us discover which indices will be nonzero
    n = x.size
    i_tau = np.where(cs - np.arange(1,n+1)*np.concatenate((s[1:],0), axis=None) >= tau)[0][0]

    # Now that we know which indices are involved, it's a very simple problem
    thresh = (cs[i_tau]-tau) / (i_tau+1)
    # Shrink x by the amount "thresh"
    return np.sign(x)*np.maximum(absx - thresh, 0)

if __name__ == '__main__':
    np.random.seed(0)
    A = np.random.rand(10, 20)
    b = np.random.rand(10)
    xcvx = cvx.Variable(20)

    # Solving BP
    bp = cvx.Problem(cvx.Minimize(cvx.norm1(xcvx)), [A*xcvx == b])
    bp.solve(verbose=False)

    # Solving LSt
    cvxopti = xcvx.value
    print('CVX: \n', cvxopti)
    tau = np.linalg.norm(cvxopti, 1)
    print('tau = {}'.format(tau))
    x0 = np.random.rand(20)
    xopti, _, _ = descend(fNg, x0, maxIterations=5e3, tol=1e-4, tolX=1e-6, linesearch=True, heavy=True, prox_g=lambda x: project_l1(x, tau), verbose=False)
    print('CVX : |{}|_1 , |A*x-b|_2={}'.format(np.linalg.norm(cvxopti, 1), np.linalg.norm(A.dot(cvxopti)-b, 2)))
    print('Mine: |{}|_1 , |A*x-b|_2={}'.format(np.linalg.norm(xopti, 1), np.linalg.norm(A.dot(xopti)-b, 2)))
    print(cvxopti.reshape(20) - xopti)

    # Run Newton
    raise ValueError("no clue where lambda is here...")
    tau_exp = tau
    tau_tmp = tau*3
    while abs(tau_tmp-tau_exp) > 0.1:
        xopti, _, _ = descend(fNg, x0, maxIterations=5e3, tol=1e-4, tolX=1e-6, linesearch=True, heavy=True, prox_g=lambda x: project_l1(x, tau_tmp), verbose=False)
        sigma, sigma_p = fNg(xopti)
        tau_tmp = tau - sigma/sigma_p
        print('new tau = ', tau_tmp)
    #endwhile
