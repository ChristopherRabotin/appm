import numpy as np
from numpy import log, exp
from numpy.random import randn
import matplotlib.pyplot as plt

from gradient import sigm, descend, check

perform_check = False

def classify(w, X, y, threshold=0.5):
    '''
    @returns: classification rate.
    '''
    num_fail = 0
    for i, email in enumerate(X):
        prediction = 1 if sigm(w.transpose().dot(email)) > threshold else -1
        if prediction != y[i]:
            num_fail += 1

    return float(num_fail)/len(X)

if __name__ == '__main__':
    plt.switch_backend('Qt4Agg')
    # load the data file
    f = open('data/hw6/spambase.data')
    X = []
    y = []
    for line in f.readlines():
        ldt = [float(v) for v in line.split(',')]
        # Change the label from {1, 0} to {1, -1}.
        if ldt[-1] == 0:
            ldt[-1] = -1
        y.append(ldt[-1])
        # Normalize the data.
        X.append([log(xij + 0.1) for xij in ldt[:-1]])
    #endfor
    # Transform X and Y to np.arrays
    X = np.array(X)
    # Shuffle X
    np.random.shuffle(X)
    train_size = 3065
    Xtrain = X[:train_size]
    y = np.array(y)
    def fNgTrain(x):
        l = lambda w: sum([log(1 + exp(-y[i]*np.dot(w.transpose(), Xtrain[i]))) for i in range(train_size)])
        nablal = lambda w: -sum([sigm(-y[i]*np.dot(w.transpose(), Xtrain[i]))*np.dot(y[i], Xtrain[i]) for i in range(train_size)])
        return l(x), nablal(x)

    def fNgFull(x):
        l = lambda w: sum([log(1 + exp(-y[i]*np.dot(w.transpose(), X[i]))) for i in range(len(X))])
        nablal = lambda w: -sum([sigm(-y[i]*np.dot(w.transpose(), X[i]))*np.dot(y[i], X[i]) for i in range(len(X))])
        return l(x), nablal(x)

    x0 = randn(57)
    if perform_check:
        hList, errors = check(fNgTrain, x0, num_points=100)
        # Show the log-log plot
        objs = plt.loglog(hList, errors, basex=2)
        plt.grid(True)
        plt.legend(objs, ['forward diff','central diff','O(h) sanity check','O(h^2)','O(h^3)'], loc='lower right')
        plt.show()

    # Plot f0 values
    L = (np.linalg.norm(X)**2)/4
    x0 = np.zeros(57)
    maxIts = 5e3
    tol = 1e-2
    tolX = 1e-2
    fixd_x, fixd_f0_values, fixd_g0_values = descend(fNgTrain, x0, linesearch=False, heavy=False, stepsize=1/L, maxIterations=maxIts, verbose=False, tol=tol, tolX=tolX)
    #fsp, = plt.semilogy(range(len(fixd_f0_values)), fixd_f0_values, label='fixed step')
    fsp = plt.semilogy(fixd_f0_values, range(len(fixd_f0_values)), label='fixed step')[0]

    lns_x, lns_f0_values, lns_g0_values = descend(fNgTrain, x0, True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    #lsp, = plt.semilogy(range(len(lns_f0_values)), lns_f0_values, label='linesearch')
    lsp = plt.semilogy(lns_f0_values, range(len(lns_f0_values)),label='linesearch')[0]

    nes_x, nes_f0_values, nes_g0_values = descend(fNgTrain, x0, True, heavy=True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    #nep, = plt.semilogy(range(len(nes_f0_values)), nes_f0_values, label='Nesterov')
    nep = plt.semilogy(nes_f0_values, range(len(nes_f0_values)),label='Nesterov')[0]   

    for name, val in [['GRADDESCENT', fixd_x[-1]], ['LINESEARCH', lns_x[-1]], ['NESTEROV', nes_x[-1]]]:
        print('[{}] Classification on training data: {}'.format(name, classify(val, X[:train_size], y[:train_size])))
        print('[{}] Classification on remaining data: {}'.format(name, classify(val, X[train_size:], y[train_size:])))

    plt.grid(True)
    plt.legend(handles=[lsp, fsp, nep], loc='upper right')
    plt.show()
