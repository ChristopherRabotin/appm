#!/usr/bin/python3
import cvxpy as cvx
import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb; pdb.set_trace()
fullset = list(csv.reader(open("winequality-white.csv", "r"), delimiter=";"))
hdrs = fullset[0]

X = np.array(fullset[1:]).astype("float")
Y = np.zeros((1, 12))

for x in range(12):
    p_vals = np.linspace(0, 1, num=1)


    # Form and solve a standard regression problem.
    Xp = X[:, x]
    Yp = Y[:, x]
    beta = cvx.Variable(1)
    print(Xp)
    # Norm 1
    costL1 = cvx.norm(Yp - Xp*beta, 1)
    probL1 = cvx.Problem(cvx.Minimize(costL1))
    probL1.solve(verbose=False)
    rsltL1 = probL1.variables()[0].value
    print("l1=", rsltL1)

    # Norm 2
    costL2 = cvx.norm(Yp - Xp*beta, 2)
    probL2 = cvx.Problem(cvx.Minimize(costL2))
    probL2.solve(verbose=False)
    rsltL2 = probL2.variables()[0].value
    print("l2=", rsltL2)

    # Plotting
    fig = plt.figure(x)
    plt.plot(p_vals, rsltL1, label='norm 1')
    plt.plot(p_vals, rsltL2, label='norm 2')
    plt.ylabel(hdrs[x])
    plt.xlabel('sample')
    plt.legend(loc='upper left')
    fig.show()
input()
