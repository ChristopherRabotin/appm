import numpy as np
from numpy import log, exp
from numpy.random import randn
import matplotlib.pyplot as plt

from gradient import sigm, descend, check

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
    train_size = 30
    Xtrain = X[:train_size]
    y = np.array(y)
    def fNgTrain(x):
        l = lambda w: sum([log(1 + exp(-y[i]*np.dot(w.transpose(), Xtrain[i]))) for i in range(train_size)])
        nablal = lambda w: -sum([sigm(-y[i]*np.dot(w.transpose(), Xtrain[i]))*np.dot(y[i], Xtrain[i]) for i in range(train_size)])
        return l(x), nablal(x)

    hList, errors = check(fNgTrain, randn(57))
    print(hList)
    # Show the log-log plot
    plt.loglog(hList, errors, basex=2)
    plt.legend('forward diff','central diff','O(h) sanity check','O(h^2)','O(h^3)','location','northwest?')
    plt.show()
