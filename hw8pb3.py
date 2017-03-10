from scipy.io import loadmat
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

def L(X):
    n1, n2 = X.size
    In1 = np.identity(n1)
    In2 = np.identity(n2)
    Dn1 = -1*np.identity(n1)
    Dn2 = -1*np.identity(n2)
    for i in range(len(Dn1)):
        for j in range(len(Dn1[i])):
            if j == i + 1:
                Dn1[i][j] = 1
    for i in range(len(Dn2)):
        for j in range(len(Dn2[i])):
            if j == i + 1:
                Dn2[i][j] = 1

    Lh = np.kron(Dn2, In2)
    Lv = np.kron(In1, Dn1)
    return (Lh, Lv)

def g(y):
    gval = []
    for yi in y:
        gval.append(np.sqrt(yi[0]**2+yi[1]**1))
    rtn = sum(gval)
    return rtn

if __name__ == '__main__':
    import h5py
    import numpy as np
    filepath = 'hw8data/SheppLogan_150x150.mat'
    Y = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        Y[k] = np.array(v)
    Y = Y['Y']

    r, c = 150, 150
    Y_noisy = np.zeros([r, c])
    for i in range(r):
        for j in range(c):
            Y_noisy[i][j] = Y[i][j]
            if np.random.uniform(0, 1) <= 0.1:
                 Y_noisy[i][j] += np.random.uniform(0, 1)

    Y_noisy_snorm = 0.5*np.sum(np.linalg.norm(np.stack(Y_noisy))**2)
    tau = 0.25*Y_noisy_snorm
    import pdb; pdb.set_trace()

    x = cvx.Variable(150)
    func = 0.5*cvx.sum_entries(cvx.norm2(cvx.hstack(Y_noisy - x))**2)
    prob = cvx.Problem(cvx.Minimize(func), constraints=[g(L(x)) <= tau])
    prob.solve(verbose=True)
    rslt = prob.variables()[0].value
    print('result = ', rslt)
