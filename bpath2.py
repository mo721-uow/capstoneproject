#BPATH2  Brownian path simulation: vectorized
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
T=1 # set plot length
N=500 # number of steps
dt=T/N # step size

dW = np.sqrt(dt) * np.random.standard_normal(N) # brownian increments
W = np.cumsum(dW) # each brownian step is the sum of brownian increments before it

plt.plot(np.arange(0, T + dt, dt), np.concatenate((np.array([0]), W)), 'r-') # plot W against t
plt.xlabel('t', fontsize=16)
plt.ylabel('W(t)', fontsize=16, rotation=0, horizontalalignment='right')
plt.tight_layout()
plt.margins(0)
plt.show()