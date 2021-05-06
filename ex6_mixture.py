import numpy as np
from random import randint
from matplotlib import pyplot as mp
import math


def gaussian(x, mu, sig):

    return 1./(np.sqrt(2.*math.pi))*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2))

# load the data
X = np.loadtxt("mixture.txt")

n=X.shape[0]
d=X.shape[1]
K = 3
#init pi
p = np.ndarray(shape=(3,1), dtype=float)
for i in range(K):
    p[i]=1/K
print(p)
#init mu
mu = np.empty([K, d])
rands = [randint(0, n-1) for p in range(K)]
print (rands)
i=0
for val in rands:
    mu[i]=X[val]
    i=i+1
print(mu)

#init sigma
sig = np.zeros((K,d,d)) 
sig[0]=np.eye(d)
sig[1]=np.eye(d)
sig[2]=np.eye(d)
print(sig)

#posterior probability gamma
posterior =np.empty([n, K])
for i in range(n):
   for j in range(K):
    posterior[i][j]=p[j]*gaussian(X[j],mu[j],sig[j])

#print(gamma) 

