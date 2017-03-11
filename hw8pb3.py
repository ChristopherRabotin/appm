from scipy.io import loadmat
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

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
    tau = 0.25
    tauV = cvx.Variable(1)

    x = cvx.Variable(r, c)

    func = (0.5*(cvx.norm2(x - Y_noisy)**2))
    prob = cvx.Problem(cvx.Minimize(func), constraints=[cvx.tv(x) <= 0.25*cvx.tv(Y_noisy), 0 <= x, x <= 1])
    prob.solve(verbose=True)
    # Save the cleaned result.
    plt.clf()
    plt.draw()
    plt.imshow(x.value, vmin=0, vmax=1)
    plt.draw()
    plt.savefig('hw8pb3-clean.png')
