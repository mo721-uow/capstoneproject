#BPATH1  Brownian path simulation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
T=1 # set plot length
N=500 # number of steps
dt=T/N # step size
dW = np.zeros(N) # create empty arrays for dW and W
W = np.zeros(N)

dW[0] = np.sqrt(dt)*np.random.standard_normal() # first approximation outside the loop
W[0] = dW[0]
for j in range(1,N-1): # skip first element and approximate
    dW[j] = np.sqrt(dt)*np.random.standard_normal()
    W[j] = W[j-1] + dW[j]

plt.plot(np.arange(0, T, dt), W, 'r-') # plot W against t
plt.xlabel('t', fontsize=16)
plt.ylabel('W(t)', fontsize=16, rotation=0)
plt.tight_layout()
plt.margins(0)
plt.show()