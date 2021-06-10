#BPATH3  Function along a Brownian path
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the seed of random
T=1 # set plot length
N=500 # number of steps
dt=T/N # step size
t = np.arange(dt, T+dt, dt) 

M = 1000 # M paths simultaneously
dW = np.sqrt(dt) * np.random.standard_normal((M,N)) # brownian increments
W = np.cumsum(dW, 1) # each brownian step is the sum of brownian increments before it
U = np.exp(np.tile(t,[M,1]) + 0.5*W) # calculate U(W(t)) for each row
Umean = U.mean(axis=0) # calculate the mean of each row

plt.plot(np.insert(t, 0, 0),np.insert(Umean, 0, 1),'b-') # plot mean over T
for j in range(0,5):
    plt.plot(np.insert(t, 0, 0), np.insert(U[j], 0, 1), 'r--')
plt.xlabel('t', fontsize=16)
plt.ylabel('U(t)', fontsize=16, rotation=0, horizontalalignment='right')
plt.legend(['mean of 1000 paths',' 5 individual paths'])
plt.margins(0)
plt.show()

averr = np.linalg.norm((Umean - np.exp(9*t/8)), np.inf)
print("averr =", averr)