# EMWEAK  Test weak convergence of Euler-Maruyama

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
emLambda = 2 # lambda
mu = .1 
Xzero = 1 
T = 1 # set the length of the plot
M = 50000 # number of paths sampled

Xem = np.zeros((5,1)) # preallocate array
for p in range(5): # take various Euler timesteps
    Dt = 2**(p-9) # delta t
    L = int(T/Dt) # L step sizes
    Xtemp = Xzero * np.ones((M,1)) # Calculate the Euler maruyama approx
    for j in range(L):
        Winc = np.sqrt(Dt) * np.random.standard_normal((M,1))
        # Winc = np.sqrt(Dt) * np.sign(np.random.standard_normal((M,1)))
        Xtemp = Xtemp + Dt*emLambda*Xtemp + mu*(np.multiply(Xtemp, Winc))
    Xem[p] = np.mean(Xtemp)
Xerr = abs(Xem - np.exp(emLambda)) # determine error

Dtvals = (np.array([2**-9, 2**-8, 2**-7, 2**-6, 2**-5]))
plt.loglog(Dtvals, Xerr, 'b*-')
plt.loglog(Dtvals, Dtvals, 'r--')
plt.axis([1*10**-3, 1*10**-1, 1*10**-4, 1])
plt.xlabel(r'$\Delta t$', fontsize=16)
plt.ylabel('| E(X(T)) - Sample average of X_L |', fontsize=20)
plt.title('emweak.py', fontsize=20)
plt.margins(0)
plt.show()

# Least squares fit of error=C*dt^q
A = np.concatenate((np.ones((5,1)), np.transpose(np.matrix(np.log(Dtvals)))), axis=1)
rhs = np.log(Xerr)
sol = np.linalg.lstsq(A, rhs, rcond=None)[0] # sol = soln x to Ax = rhs
q = sol[1]
resid = np.linalg.norm(A*sol - rhs)
print("q =", q, "resid =", resid)