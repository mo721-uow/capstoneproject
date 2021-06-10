# CHAIN  Test stochastic Chain Rule

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
alpha = 2 # problem parameter
beta = 1 # problem parameter
T = 1 # duration of graph
N = 200 # number of steps
dt = T/N # step size
Xzero = 1 # xzero for solution 1
Xzero2 = 1/np.sqrt(Xzero) # xzero for solution 2

Dt = dt # EM steps of size Dt = dt
Xem1 = np.zeros((N + 1, 1)) # create array for soln 1
Xem2 = np.zeros((N + 1, 1)) # create array for soln 2
Xem1[0] = Xzero # set xzero for soln 1
Xem2[0] = Xzero2 # set xzero for soln 2

# calculate two EM approximations simultaneously
Xtemp1 = Xzero
Xtemp2 = Xzero2
for j in range(N):
  Winc = np.sqrt(dt)*np.random.standard_normal()
  f1 = (alpha - Xtemp1)
  g1 = beta * np.sqrt(abs(Xtemp1))
  Xtemp1 = Xtemp1 + Dt*f1 + Winc*g1
  Xem1[j+1] = Xtemp1
  f2 = (4*alpha - beta**2) / (8*Xtemp2) - Xtemp2/2
  g2 = beta/2
  Xtemp2 = Xtemp2 + Dt*f2 + Winc*g2
  Xem2[j+1] = Xtemp2

# plot the solutions
plt.plot(np.arange(0, T + Dt, Dt), np.sqrt(Xem1), 'b-', label='Direct Solution')
plt.plot(np.arange(0, T + Dt, Dt), Xem2, 'ro', fillstyle='none', label='Solution via Chain Rule')
plt.xlabel('t', fontsize=16)
plt.ylabel('V(X)', fontsize=16, rotation=0, horizontalalignment='right')
plt.tight_layout()
plt.margins(0)
plt.show()

Xdiff = np.linalg.norm(np.sqrt(Xem1) - Xem2, np.inf)
print("Xdiff =", Xdiff)