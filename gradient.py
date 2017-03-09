import numpy as np
from numpy import exp, logspace, log10, zeros, sqrt, dot
from numpy.linalg import norm
from numpy.random import randn

def sigm(x):
    return 1 / (1+exp(-x))

def check(funcNgrad, x0, scaling=1, num_points=8, verbose=True):
    '''
    Mostly a conversion from https://github.com/stephenbeckr/convex-optimization-class/blob/master/HW6/gradientCheck.m.
    @parameter funcNgrad: a function handler which returns both the function and its first order gradient
    @parameter x0: the x0 at which to evaluate funcNgrad (will be called as `funcNgrad(x0)`) (as np.array)
    @parameter scaling: scales the step size
    @parameter num_points: number of points to use for values of h
    @return: an array of the list of values of h and the errors.
    '''
    n = x0.size
    f0, g0 = funcNgrad(x0)
    hfinal  = log10(scaling)/2+1
    hList = logspace(7+hfinal, hfinal, num_points)
    if verbose:
        print('h\t\tForward diff\tCentral diff\t1st order Taylor\t2nd order Taylor\t3rd order Taylor')
    errors = []

    for h in hList:
        g = zeros((n, 1))
        e = zeros((n, 1))
        gc = zeros((n, 1))
        for i in range(n):
            e[i] = 1 # Used to ignore everything in the multiplication but the current i
            f1 = funcNgrad(x0 + h*e)[0]
            g[i] = (f1 - f0)/h
            gc[i] = (f1 - funcNgrad(x0 - h*e)[0])/(2*h);
            e[i] = 0

        er_fd = norm(g - g0)/norm(g0)
        er_cd = norm(gc - g0)/norm(g0)

        nReps = 5
        Taylor1, Taylor2, err3 = 0, 0, 0
        for rep in range(nReps):
            x1  = x0 + h*randn(x0.size)/sqrt(x0.size)*norm(x0);
            f1, g1  = funcNgrad(x1);

            Taylor1 = Taylor1 + abs(f0 - f1) # O(h)
            Taylor2 = Taylor2 + abs(f0 + dot(g0, x1-x0) - f1) # O(h^2)
            err3  = err3 + abs(dot(g0+g1, x1-x0) - 2*(f1-f0)) # at least o(h^2)
        Taylor1 /= nReps
        Taylor2 /= nReps
        err3 /= nReps

        if verbose:
            print('%.1e\t\t%.1e\t\t%.1e\t\t%.1e\t\t\t%.1e\t\t\t%.1e' % (h, er_fd, er_cd, Taylor1, Taylor2, err3 ))

        errors.append([er_fd, er_cd, Taylor1, Taylor2, err3])

def descend(funcNgrad, value, linesearch=True, stepsize=1, c=1e-4, rho=0.9, maxIterations=1e4, tol=1e-6, tolX=1e-6, verbose=True):
    '''
    Mostly a conversion from https://github.com/stephenbeckr/convex-optimization-class/blob/master/HW6/gradientDescent.m.
    @parameter funcNgrad: a function handler which returns both the function and its first order gradient
    @parameter value: the value at which to evaluate funcNgrad (will be called as `funcNgrad(value)`)
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
    for dit in range(int(maxIterations)):
        fVal, gVal = funcNgrad(value)
        if linesearch:
            if dit > 0:
                t*=2 # Be aggressive
            #endif
            # Search for a better value
            lnsCnt = 0
            while  funcNgrad(value - t*gVal)[0] > fVal - t*c*norm(gVal)**2:
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
            print('[%4d] f=%.2e\t|g|=%.2e\tlinesearch steps: %2d' % (dit, fVal, norm(gVal), lnsCnt), flush=True)
        #endif
        value = t*gVal
    #endfor
    if objConverged:
        print("Convergence in objective")
    elif varConverged:
        print("Convergence in variable")
    else:
        print("WARNING: No convergence")
    #endif
    return value

if __name__ == '__main__':
    def fNg(x):
        f = lambda x: x**2
        g = lambda x: 2*x
        return f(x), g(x)

    check(fNg, np.array([5]))
