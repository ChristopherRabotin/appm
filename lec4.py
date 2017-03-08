#!/usr/bin/python3
import cvxpy as cvx
import numpy as np

A = np.array([ [1,6,11,5,10,4,9,3,8,2],
                [2,7,1,6,11,5,10,4,9,3],
                [3,8,2,7,1,6,11,5,10,4],
                [4,9,3,8,2,7,1,6,11,5],
                [5,10,4,9,3,8,2,7,1,6]])

y = np.array([1,2,3,4,5])
x = cvx.Variable(10)

pbn = 4

# Using if instead of elif because I need the variable of pb1 in pb4.

if pbn == 1 or pbn == 4:
    pb = cvx.Problem(cvx.Minimize(cvx.norm2(x)), [cvx.norm2(A*x - y) <= 0.1])
    pb.solve()
    print(pb.status, pb.value, "should be 0.294216")
if pbn == 2:
    pb = cvx.Problem(cvx.Minimize(cvx.norm2(x)**2), [cvx.norm2(A*x - y) <= 0.1])
    pb.solve()
    print(pb.status, pb.value, "should be 0.0865633")
if pbn == 3:
    pb = cvx.Problem(cvx.Minimize(cvx.norm1(x)), [cvx.norm2(A*x - y) <= 0.1])
    pb.solve()
    print(pb.status, pb.value, "should be 0.787669")
if pbn == 4:
    l = pb.constraints[0]#.dual_variable
    pb = cvx.Problem(cvx.Minimize(cvx.norm(x, 2)) + l*cvx.norm(A*x - y))
    pb.solve()
    print(pb.status, pb.value, "should be 0.294216")
