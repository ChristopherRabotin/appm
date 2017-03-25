import numpy as np
from numpy import log, exp
from numpy.random import randn
import matplotlib.pyplot as plt

from gradient import sigm, descend, check
from scipy.io import loadmat

perform_check = False
use_scipy = True
pb_num = 1

if use_scipy:
    from scipy.optimize import fmin

fmin_x, fmin_values = [], []

def prox2(x, t, g):
    return np.sign(x)*max(np.linalg.norm(x) - t*5, 0)

def classify(w, X, y, threshold=0.5):
    '''
    @returns: classification rate.
    '''
    num_correct = 0
    y_prediction = []
    for i, email in enumerate(X):
        classification = 1 if sigm(np.inner(w, email)) > threshold else -1
        y_prediction.append(classification)
        if classification == y[i]:
            num_correct+=1

    y_prediction = np.array(y_prediction)
    return sum(y*y_prediction)/(2*len(X)) + 0.5

if __name__ == '__main__':
    #plt.switch_backend('Qt4Agg')
    plt.switch_backend('agg')
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

    def fNgGeneric(x, Xset, Yset):
        n = y_train.size
        l = lambda w: sum([log(1 + exp(-Yset[i]*np.inner(w, Xset[i]))) for i in range(n)])
        nablal = lambda w: -sum([sigm(-Yset[i]*np.inner(w, Xset[i]))*np.dot(Yset[i], Xset[i]) for i in range(n)])
        return l(x), nablal(x)

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
    x0 = np.zeros(57)
    maxIts = 5e3
    tol = 1e-2
    tolX = 1e-2

    print('[INFO] Starting line search gradient descent (no g)')
    lns_x0, lns_f0_values0, lns_g0_values0 = descend(fNgTrain, x0, linesearch=True, heavy=True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    lsp0 = plt.semilogy(lns_f0_values0, range(len(lns_f0_values0)),label='linesearch (no g)')[0]

    print('[INFO] Starting line search gradient descent (with g)')
    lns_x1, lns_f0_values1, lns_g0_values1 = descend(fNgTrain, x0, linesearch=False, heavy=True, maxIterations=maxIts,verbose=False, tol=tol, tolX=tolX)
    lsp1 = plt.semilogy(lns_f0_values1, range(len(lns_f0_values1)),label='linesearch (with g)')[0]

    for name, val in [['LINESEARCH 0', lns_x0[-1]], ['LINESEARCH 1', lns_x1[-1]]]:
        print('[{}] Classification on training data: {}'.format(name, classify(val, X_train, y_train)))
        print('[{}] Classification on remaining data: {}'.format(name, classify(val, X_test, y_test)))

    plt.grid(True)
    plt.legend(handles=[lsp0, lsp1], loc='upper right')
    plt.show()
    plt.draw()
    plt.savefig('hw8pb2-0.png')

    # Plot the weights
    p0 = plt.plot(range(len(lns_x0[-1])), lns_x0[-1], label='linesearch (no g)')[0]
    p1 = plt.plot(range(len(lns_x1[-1])), lns_x1[-1], label='linesearch (with g)')[0]
    plt.grid(True)
    plt.legend(handles=[p0, p1], loc='upper left')
    plt.show()
    plt.draw()
    plt.savefig('hw8pb2-1.png')
