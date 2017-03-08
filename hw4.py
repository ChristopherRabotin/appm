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
    @returns the convolved signal (with shift removed!)
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
    obj = cvx.Minimize(cvx.norm(x, 1))
    y = blur(x_signal)
    # Add the noise
    y = [val + np.random.normal(0, sigma) for val in y]
    constraints = [cvx.norm(B*x-y, 2) <= esp]

    prob = cvx.Problem(obj, constraints)
    prob.solve()

    print("Problem status: ", prob.status)
    print("Optimal value:  ", prob.value)

    # Plotting the original signal
    X = np.linspace(0, N, N)
    pxsig = plt.plot(X, x_signal, 'o', mfc='none', label='true signal')
    py = plt.plot(X, y, label='noisy blurred signal')
    pxest = plt.plot(X, x.value, 'x', label='estimated signal')
    plt.legend()
    plt.show()
