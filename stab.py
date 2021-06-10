# STAB  Mean-square and asymptotic stability test for E-M

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
ltype = ['b-', 'r--', 'm-.'] # different plot types
legend = ['$\Delta t = 1$','$\Delta t = 1/2$','$\Delta t = 1/4$'] # different plot labels
M=50000 # number of brownian paths
Xzero = 1 # X(0)

fig, (ax1, ax2) = plt.subplots(2) # plot has 2 subplots

# Mean square
T = 20 # plot length
stabLambda = -3 #lambda
mu = np.sqrt(3) 
for k in range(3): # repeat 3 times
  Dt = 2**(-k) # delta t = 1, 1/2, 1/4
  N = int(T/Dt) # number of steps
  Xms = np.zeros((N + 1, 1)) # create empty array to store plot
  Xms[0] = Xzero # set Xzero
  Xtemp = Xzero * np.ones((M, 1))
  for j in range(N): # calculate mean squared estimate for em approximation
    Winc = np.sqrt(Dt)*np.random.standard_normal((M,1))
    Xtemp = Xtemp + Dt*stabLambda*Xtemp + np.multiply(mu*Xtemp, Winc)
    Xms[j + 1] = np.mean(np.power(Xtemp, 2)) # mean-squared estimate
  ax1.semilogy(np.arange(0, T + Dt, Dt), Xms, ltype[k], linewidth = 2, label = legend[k]) # plot log

ax1.set_title(r'Mean-Square: $\lambda = -3, \mu = \sqrt{3}$', fontsize = 16)
ax1.set_ylabel(r'$E[X^2]$', fontsize = 12)
ax1.axis([0,T,1e-20,1e+20])
ax1.set_yticks([1e-20, 1e0, 1e20])
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ax1.legend()

# Asymptotic: a single path
T = 500 # plot length
stabLambda = 0.5 # lambda
mu = np.sqrt(6)
for k in range(3): # repeat 3 times
  Dt = 2**(-k) # delta t = 1, 1/2, 1/4
  N = int(T/Dt) # number of steps
  Xemabs = np.zeros((N + 1, 1)) # create empty array to store plot
  Xtemp = Xzero
  for j in range(N):
     Winc = np.sqrt(Dt)*np.random.standard_normal((M, 1))
     Xtemp = Xtemp + Dt*stabLambda*Xtemp + mu*Xtemp*Winc
     XtempAbs = np.sqrt(np.sum(np.square(Xtemp)))
     Xemabs[j+1] = XtempAbs # calculate asymptotic estimate
  Xemabs[0] = Xzero # set Xzero
  ax2.semilogy(np.arange(0, T + Dt, Dt), Xemabs, ltype[k], linewidth = 2, label = legend[k]) # plot log

ax2.set_title(r'Single Path: $\lambda = 1/2, \mu = \sqrt{6}$', fontsize=16)
ax2.set_ylabel(r'$|X|$', fontsize=16)
ax2.axis([0,T,1e-50,1e+100])
ax2.set_yticks([1e-50, 1e0, 1e50, 1e100])
ax2.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
ax2.legend()

plt.tight_layout()
plt.show()