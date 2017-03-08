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

def BPrime(z):
    filterHPadded = np.zeros(len(z))
    filterHPadded[:len(filterH)] = filterH
    return np.fft.ifft(np.multiply(np.fft.fft(z), np.fft.fft(filterHPadded)))

def BStar(z):
    filterHPadded = np.zeros(len(z))
    filterHPadded[:len(filterH)] = filterH
    return np.fft.ifft(np.multiply(np.fft.fft(z), np.fft.fft(filterHPadded).conj()))

if __name__ == '__main__':
    N = 100
    sigma = 0.04
    esp = sigma*sqrt(N)

    B = funcAsMatrix(blur, N)
    # Minimization problem
    x = cvx.Variable(100)
    y = blur(x_signal)
    yP = BPrime(x_signal)


    # Some tests
    sigma = 0.04
    samples = 10
    axis_x = []
    axis_y = []
    for N in range(50, 1050, 50):
        axis_x += [N]
        abssum = 0
        for _ in range(samples):
            randX = np.array([np.random.normal(0, sigma) for _ in range(N)])
            randY = np.array([np.random.normal(0, sigma) for _ in range(N)])
            diff = abs(np.dot(randX, BStar(randY)) - np.dot(BPrime(randX), randY))
            if diff > 1e-8:
                raise ValueError("invalid absolute difference of {} for N={}".format(diff, N))
            abssum += diff
        axis_y += [abssum/samples]
    # Plot the histogram of errors
    plt.bar(axis_x, axis_y, 10,  label='average errors with {} samples vs. vector size'.format(samples))
    plt.legend()
    plt.show()
    print("OK")
