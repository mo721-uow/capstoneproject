# EM Euler-Maruyama method on linear SupportsIndex

import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
emLambda = 2 #lambda
mu = 1
Xzero = 1 # start value X0
T = 1 # plot length
N = 2**8 # number of steps
dt = 1/N # step size
dW = np.sqrt(dt) * np.random.standard_normal(N) # brownian steps
W = np.cumsum(dW) # each brownian step is the sum of brownian increments before it

# calculate and plot true value of X
Xtrue = Xzero * np.exp((emLambda-.5*(mu**2)) * (np.arange(dt, T + dt, dt) + mu * W))
plt.plot(np.arange(0, T + dt, dt),np.concatenate(([Xzero], Xtrue)), 'm-')

# calculate the EM approximation
R = 4
Dt = R*dt # em step size
L = int(N/R) # L em steps
Xem = np.zeros(L) # create empty array for em values
Xtemp = Xzero
for j in range(L): # calculate em method
    Winc = np.sum(dW[R*j:R*(j+1)])
    Xtemp = Xtemp + Dt*emLambda*Xtemp + mu*Xtemp*Winc
    Xem[j] = Xtemp

# plot em method over time
plt.plot(np.arange(0, T + Dt, Dt),np.concatenate(([Xzero], Xem)),'r--*')
plt.xlabel('t', fontsize=16)
plt.ylabel('X', fontsize=16, rotation=0)
plt.tight_layout()
plt.margins(0)
plt.show()

emerr = abs(Xem[-1] - Xtrue[-1]) # calculate error from true solution
print("emerr =", emerr)