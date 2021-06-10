# MILSTRONG  Test strong convergence of Milstein

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
r = 2 # set constant
K = 1 # set constant
beta = .25 # set constant
Xzero = .5 # set constant
T = 1 # set length of plot
N = 2**11  # number of intervals
dt = T/N # interval size
M = 500 # number of brownian tests
R = np.matrix('1; 16; 32; 64; 128') # multiples of delta t to test

dW = np.sqrt(dt)*np.random.standard_normal((M, N)) # brownian increments
Xmil = np.zeros((M, 5)) # prefill arrary for millstein
for p in range(5): # for each multiple of delta t
    Dt = int(R[p]) * dt # scale dt for delta t
    L = int(N/int(R[p])) # L time steps for delta t
    Xtemp = Xzero * np.ones((M,1))
    for j in range(L):
        Winc = np.zeros((500, 1)) # prefill array 
        start = int(R[p]) * j # sum function start
        end = int(R[p]) * (j+1) # sum function end
        for row in range(Winc.shape[0]): # mimic sum function of matlab
            for col in range(end-start):
                Winc[row] = Winc[row] + dW[row][start+col]
        Xtemp = Xtemp + np.multiply(Dt*r*Xtemp, (K-Xtemp)) + beta*np.multiply(Xtemp, Winc) + 0.5*(beta**2)*np.multiply(Xtemp, np.power(Winc, 2) - Dt)
    for col in range(Xmil.shape[0]):
        Xmil[col][p] = float(Xtemp[col])

# referencing single rows of numpy matrices doesnt behave well, move rows to new matrices.
Xref = np.zeros((Xmil.shape[0], 1)) # matrice for "true" soln
for row in range(Xref.shape[0]):
    Xref[row] = Xmil[row][0]
Xmil2 = np.zeros((Xmil.shape[0],4)) # matrice for approximation
for row in range(Xref.shape[0]):
    for col in range(4):
        Xmil2[row][col] = Xmil[row][col + 1]
Xerr = np.abs(Xmil2 - np.tile(Xref, (1,4))) # matrice of error for each approx

Dtvals = dt*np.matrix('16; 32; 64; 128')

plt.loglog(Dtvals, np.mean(Xerr, axis=0), 'b*-')
plt.loglog(Dtvals, Dtvals, 'r--')
plt.axis([1*10**-3, 1*10**-1, 1*10**-4, 1])
plt.xlabel(r'$\Delta t$', fontsize=16)
plt.ylabel('Sample average of | X(T) - X_L |', fontsize=20)
plt.title('milstrong.py', fontsize=20)
# plt.margins(0)
plt.show()

# Least squares fit of error=C*Dt^q 
A = np.concatenate((np.ones((4,1)), np.log(Dtvals)), axis = 1)
rhs = np.transpose(np.matrix(np.log(np.mean(Xerr, axis=0))))
sol = np.linalg.lstsq(A, rhs, rcond=None)[0] # sol = soln x to Ax = rhs
q = sol[1]
resid = np.linalg.norm(A*sol - rhs)
print("q =", q, "resid =", resid)