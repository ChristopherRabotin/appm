#!/usr/bin/python3
import csv
import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

fullset = list(csv.reader(open("winequality-white.csv", "r"), delimiter=";"))
hdrs = fullset[0]

dataset = []
max_quality = 8
min_quality = 4

# Some filtering
for lno, sample in enumerate(fullset[1:]):
    quality = float(sample[-1])
    if quality > max_quality or quality < min_quality:
        print('skipping line', lno, 'quality:', quality)
    else:
        dataset.append(sample)

DATA = np.array(dataset).astype("float")
A = DATA[:, :11]
B = DATA[:, 11]
p_vals = np.linspace(1, A.shape[0], num=A.shape[0])


# Form and solve a standard regression problem.
x = cvx.Variable(A.shape[1])

# Norm 1
costL1 = cvx.norm(B - A*x, 1)
probL1 = cvx.Problem(cvx.Minimize(costL1))
probL1.solve(verbose=True)
rsltL1 = probL1.variables()[0].value

# Norm 2
costL2 = cvx.norm(B - A*x, 2)
probL2 = cvx.Problem(cvx.Minimize(costL2))
probL2.solve(verbose=True)
rsltL2 = probL2.variables()[0].value

print('l1 = ', rsltL1)
print('l2 = ', rsltL2)

plt.plot(p_vals, B, 'g.', label='data')
plt.plot(p_vals, A*rsltL1, 'r.', label='l1')
plt.plot(p_vals, A*rsltL2, 'y.', label='l2')
plt.ylabel('quality')
plt.xlabel('sample number')
plt.legend(loc='upper right')
plt.title('Linear regressions (l1 and l2) -- outliers removed')
plt.show()
