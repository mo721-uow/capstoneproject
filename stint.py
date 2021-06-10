#STINT  Approximate stochastic integrals

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100) # set the state of random
T=1 # set path length
N=500 # number of steps
dt=T/N # step size

dW = np.sqrt(dt) * np.random.standard_normal(N) # brownian increments
W = np.cumsum(dW) # each brownian step is the sum of brownian increments before it

ito = np.sum(np.multiply(np.concatenate(([0], W[:-1])), dW)) # ito by defn
print("ito =", ito)
strat = np.sum(np.multiply(0.5*(np.concatenate(([0], W[:-1])) + W) + 0.5*np.sqrt(dt)*np.random.standard_normal((1,N)), dW)) # strat by defn
print("strat =", strat)

itoerr = abs(ito - .5*(W[-1]**2 - T)) # calculates error of ito approximation
print("itoerr =", itoerr)
straterr = abs(strat - 0.5*W[-1]**2) # calculates error of strat approximation
print("straterr =", straterr)