from numpy import exp
from numpy.linalg import norm

def sigm(x):
    return 1 / (1+exp(-x))

def check():
    pass

def descend(fNgrad, value, linesearch=True, stepsize=1, c=1e-4, rho=0.9, maxIterations=1e4, tol=1e-6, tolX=1e-6, verbose=True):
    '''
    Mostly a conversion from https://github.com/stephenbeckr/convex-optimization-class/blob/master/HW6/gradientDescent.m.
    @parameter fNgrad: a function handler which returns both the function and its first order gradient
    @parameter value: the value at which to evaluate fNgrad (will be called as `fNgrad(value)`)
    @parameter linesearch: set to True to perform a back tracking linesearch
    @parameter c: backtracking parameter for the gradient
    @parameter rho: backtracking parameter for t
    @parameter maxIterations: maximum number of iterations before giving up
    @parameter tol: relative tolerance to the object
    @parameter tolX: relative tolerance in ||x_{k} - x_{k-1}||
    '''
    t = stepsize
    objConverged = False
    varConverged = False
    lnsCnt = 0
    for dit in range(maxIterations):
        fVal, gVal = fNgrad(value)
        if linesearch:
            if dit > 0:
                t*=2 # Be aggressive
            #endif
            # Search for a better value
            lnsCnt = 0
            while  fNgrad(value - t*gVal)[0] > fVal - t*c*norm(g)**2:
                lnsCnt += 1
                t *= rho
            #endwhile
        #endif
        if norm(t*gVal)/norm(value) < tolX:
            objConverged = True
            break
        #endif
        if dit > 0 and abs(fVal - float('inf')) < tol:
            varConverged = True
            break
        #endif
        if verbose:
            print('[%4d] f=%.2e\t|g|=%.2e\tlinesearch steps: %2d' % (dit, fVal, norm(gVal), lnsCnt))
        #endif
    #endfor
    if objConverged:
        print("Convergence in objective")
    elif varConverged:
        print("Convergence in variable")
    else:
        print("WARNING: No convergence")
    #endif
