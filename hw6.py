import numpy as np
from numpy import log, exp
from numpy.random import randn
import matplotlib.pyplot as plt

from gradient import sigm, descend, check

perform_check = False
use_scipy = True
use_matfile = True

from scipy.io import loadmat

if use_scipy:
    from scipy.optimize import fmin

fmin_x, fmin_values = [], []

def fmin_callback(*args, **kwargs):
    x = args[0]
    fmin_x.append(x)
    fmin_values.append(fNgTrain(x)[0])

def sign(x):
    if x < 0:
        return -1
    return 1

def classify0(w, X, y, threshold=0.5):
    '''
    @returns: classification rate.
    '''
    num_fail = 0
    for i, email in enumerate(X):
        prediction = 1 if sigm(np.inner(w, email)) > threshold else -1
        if prediction != y[i]:
            num_fail += 1

    return float(num_fail)/len(X)

def classify(w, X, y, threshold=0.5):
    '''
    @returns: classification rate.
    '''
    y_prediction = []
    for i, email in enumerate(X):
        y_prediction.append(1 if sigm(np.inner(w, email)) > threshold else -1)

    y_prediction = np.array(y_prediction)
    return sum(y*y_prediction)/(2*len(X)) + 0.5

if __name__ == '__main__':
    #plt.switch_backend('Qt4Agg')
    plt.switch_backend('agg')
    X = []
    y = []
    X_train, X_test, y_train, y_test = None, None, None, None
    if use_matfile:
        # Loading code from Manuel
        data_mat = loadmat('data/hw6/spambase.mat')
        # Test data
        X_test = np.array(data_mat['Xtest'])
        y_test = np.array(data_mat['ytest'], dtype=np.int8)
        # Each row of X_train contains a case of the 57 inputs
        # Each input in y_train corresponds to isSpam (1) or isNotSpam (0)
        X_train = np.array(data_mat['Xtrain'])
        y_train = np.array(data_mat['ytrain'], dtype=np.int8)
        n = y_train.size
        p = 57
        ### Transforming 0 -> -1
        y_train = np.squeeze(2*y_train - 1)
        y_test = np.squeeze(2*y_test - 1)
        ### Pre-process
        X_train = np.log(X_train + 0.1)
        X_test = np.log(X_test + 0.1)
    else:
        # load the data file
        f = open('data/hw6/spambase.data')
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
        y = np.array(y)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

    #endif

    def fNgGeneric(x, Xset, Yset):
        n = y_train.size
        l = lambda w: sum([log(1 + exp(-Yset[i]*np.inner(w, Xset[i]))) for i in range(n)])
        nablal = lambda w: -sum([sigm(-Yset[i]*np.inner(w, Xset[i]))*np.dot(Yset[i], Xset[i]) for i in range(n)])
        return l(x), nablal(x)

        #l = lambda w: sum([log(1 + exp(-Yset[i]*np.dot(w.transpose(), Xset[i]))) for i in range(rgn)])
        #nablal = lambda w: -sum([sigm(-Yset[i]*np.dot(w.transpose(), Xset[i]))*np.dot(Yset[i], Xset[i]) for i in range(rgn)])
        #return l(x), nablal(x)

    def fNgTrain(x):
        return fNgGeneric(x, X_train, y_train)

    def fNgTest(x):
        return fNgGeneric(x, X_test, y_test)

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
    maxIts = 2e3
    tol = 1e-3
    tolX = 1e-3

    print('[INFO] Starting fixed step gradient descent')
    fixd_x, fixd_f0_values, fixd_g0_values = descend(fNgTrain, x0, linesearch=False, heavy=False, stepsize=1/L, maxIterations=maxIts, verbose=False, tol=tol, tolX=tolX)
    fsp = plt.semilogy(fixd_f0_values, range(len(fixd_f0_values)), label='fixed step')[0]

    print('[INFO] Starting line search gradient descent')
    lns_x, lns_f0_values, lns_g0_values = descend(fNgTrain, x0, linesearch=True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    lsp = plt.semilogy(lns_f0_values, range(len(lns_f0_values)),label='linesearch')[0]

    print('[INFO] Starting line search Nesterov (heavy ball)')
    nes_x, nes_f0_values, nes_g0_values = descend(fNgTrain, x0, linesearch=True, heavy=True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    nep = plt.semilogy(nes_f0_values, range(len(nes_f0_values)),label='Nesterov')[0]

    print('[INFO] Starting fixed step Nesterov (heavy ball)')
    nes2_x, nes2_f0_values, nes2_g0_values = descend(fNgTrain, x0, linesearch=False, heavy=True, stepsize=1/L, maxIterations=maxIts, verbose=False, tol=tol, tolX=tolX)
    nep2 = plt.semilogy(nes2_f0_values, range(len(nes2_f0_values)), label='Nesterov fixed')[0]


    print('[INFO] Starting scipy\'s fmin')
    # Scipy fmin
    def sfunc(x):
        return fNgTrain(x)[0]
    fmin(sfunc, x0, xtol=tolX, ftol=tol, maxiter=maxIts, full_output=True, callback=fmin_callback)
    fmp = plt.semilogy(fmin_values, range(len(fmin_values)),label='Scipy fmin')[0]

    for name, val in [['GRADDESCENT', fixd_x[-1]], ['LINESEARCH', lns_x[-1]], ['NESTEROV', nes_x[-1]], ['NESTEROV 2', nes2_x[-1]], ['FMIN', fmin_x[-1]]]:
        print('[{}] Classification on training data: {}'.format(name, classify(val, X_train, y_train)))
        print('[{}] Classification on remaining data: {}'.format(name, classify(val, X_test, y_test)))

    plt.grid(True)
    #plt.legend(handles=[lsp, nep, fmp], loc='upper right')
    plt.show()
    plt.draw()
    plt.savefig('hw6.png')
