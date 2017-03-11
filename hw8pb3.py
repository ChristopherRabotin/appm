from scipy.io import loadmat
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

def L(X):
    n1, n2 = 15, 15
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
    return np.vstack((Lh, Lv))

def g(y):
    gval = []
    for yi in y:
        gval.append(np.sqrt(yi[0]**2+yi[1]**2))
    return np.sum(np.array(gval))

if __name__ == '__main__':
    import h5py
    import numpy as np
    filepath = 'hw8data/SheppLogan_150x150.mat'
    Y = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        Y[k] = np.array(v)
    Y = Y['Y']

    # Save loaded image to check
    plt.imshow(Y, vmin=0, vmax=1)
    plt.draw()
    plt.savefig('hw8pb3-normal.png')


    r, c = 150, 150
    Y_noisy = np.zeros([r, c])
    noise_points = 0
    for i in range(r):
        for j in range(c):
            Y_noisy[i][j] = Y[i][j]
            if np.random.uniform(0, 1) <= 0.1:
                 noise_points += 1
                 Y_noisy[i][j] += np.random.uniform(0, 1)

    print('added {} noise points (should be {})'.format(noise_points, r*c*0.1))
    Y_noisy_snorm = 0.5*np.sum(np.linalg.norm(np.stack(Y_noisy))**2)
    # Save noisy image
    plt.clf()
    plt.draw()
    plt.imshow(Y_noisy, vmin=0, vmax=1)
    plt.draw()
    plt.savefig('hw8pb3-noisy.png')
    tau = 0.25*Y_noisy_snorm
    tauV = cvx.Variable(1)

    x = cvx.Variable(r, c)
    z = cvx.Variable(450, 225)

    func = (0.5*cvx.sum_entries(cvx.norm2(Y_noisy - x)**2))
    prob = cvx.Problem(cvx.Minimize(func), constraints=[L(x) == z, g(z) <= tauV, tauV == tau, 0 <= x, x <= 1])
    prob.solve(verbose=True)
    # Save the result.
    plt.clf()
    plt.draw()
    plt.imshow(Y_noisy - x, vmin=0, vmax=1)
    plt.draw()
    plt.savefig('hw8pb3-fixed.png')
