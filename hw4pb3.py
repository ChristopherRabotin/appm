 #!/usr/bin/python3

from math import exp, sqrt

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

x_signal = [0] * 100
# Specific values are non nil:
x_signal[9] = 1
x_signal[12] = -1
x_signal[49] = 0.3
x_signal[69] = -0.2

filterH = []
for x in range(-2,3):
    filterH.append(exp(-(x**2)/2))

def blur(sig):
    '''
    @returns the convolved signal
    '''
    return np.convolve(sig, filterH)[2:-2]

def funcAsMatrix(func, N):
    '''
    @param func: the function
    @param N: the dimension
    I prefer this name to `implicit2explicit`
    '''
    sig = lambda x: [0 if i != x else 1 for i in range(N)]
    matcols = []
    for i in range(N):
        matcols.append(func(sig(i)))
    return np.matrix(matcols)

if __name__ == '__main__':
    N = 100
    sigma = 0.04
    esp = sigma*sqrt(N)

    B = funcAsMatrix(blur, N)
    # Minimization problem
    x = cvx.Variable(100)
    y = blur(x_signal)
    # Add the noise
    y = [val + np.random.normal(0, sigma) for val in y]
    constraints = [cvx.norm(B*x-y, 2) <= esp]

    pb2 = cvx.Problem(cvx.Minimize(cvx.norm(x, 1)), constraints)
    pb2.solve()

    # Let's solve pb3 using the dual value from pb2.
    xPrime = cvx.Variable(100)
    pb3 = cvx.Problem(cvx.Minimize(cvx.norm(xPrime, 1) + constraints[0].dual_value*(cvx.norm(B*xPrime-y, 2))**2))
    pb3.solve()

    # Plotting the original signal
    X = np.linspace(0, N, N)
    plt.plot(X, x_signal, 'o', mfc='none', label='true signal')
    plt.plot(X, y, label='noisy blurred signal')
    plt.plot(X, x.value, 'x', label='estimated signal (model 1)')
    plt.plot(X, xPrime.value, '.', label='estimated signal (model 2)')
    plt.legend()
    plt.show()
    # Difference in estimates
    plt.plot(X, abs(x.value-xPrime.value), 'x', label='absolute difference in estimates of signal')
    plt.legend()
    plt.show()
