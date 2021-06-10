# EMStrong Test strong convergence of Euler-Maruyama

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
emLambda = 2 # lambda
mu = 1 
Xzero = 1 
T = 1 # set the length of the plot
N = 2 ** 9 # number of intervals
dt = T/N # interval size
M = 1000 # number of paths sampled

Xerr = np.zeros((M, 5)) # preallocate array
for s in range(M): # compute over M Brownian paths
    dW = np.sqrt(dt) * np.random.standard_normal(N) # Brownian steps
    W = np.cumsum(dW) # each brownian step is the sum of brownian increments before it 
    Xtrue = Xzero*np.exp((emLambda-.5*mu**2) + mu*W[-1])
    for p in range(5): # for each multiple of delta t, EM approximation
        R = 2 ** p
        Dt = R*dt
        L = int(N/R) # L em steps of size Dt = R*dt
        Xtemp = Xzero
        for j in range(L):
            Winc = np.sum(dW[R*j:R*(j+1)])
            Xtemp = Xtemp + Dt*emLambda*Xtemp + mu*Xtemp*Winc
        Xerr[s,p] = abs(Xtemp - Xtrue) # stores the error at t=1
        
Dtvals = dt*(np.array([1, 2, 4, 8, 16]))
plt.loglog(Dtvals, np.mean(Xerr, axis=0), 'b*-')
plt.loglog(Dtvals, np.power(Dtvals, np.array([.5, .5, .5, .5, .5])), 'r--')
plt.axis([1*10**-3, 1*10**-1, 1*10**-4, 1])
plt.xlabel(r'$\Delta t$', fontsize=16)
plt.ylabel('Sample average of | X(T) - X_L |', fontsize=20)
plt.title('emstrong.py', fontsize=20)
plt.margins(0)
plt.show()

# Least squares fit of error=C*Dt^q
A = np.concatenate((np.ones((5,1)), np.transpose(np.matrix(np.log(Dtvals)))), axis=1)
rhs = np.log(np.transpose(np.matrix(np.mean(Xerr, axis = 0))))
sol = np.linalg.lstsq(A, rhs, rcond=None)[0] # sol = soln x to Ax = rhs
q = sol[1]
resid = np.linalg.norm(A*sol - rhs)
print("q =", q, "resid =", resid)