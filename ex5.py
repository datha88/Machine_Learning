import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os
import scipy.sparse.linalg as sla
from random import randint

rows = 166
columns = 77760
samples = 2

def load_images(path):
    dirs = os.listdir( path )
    X = np.empty([len(dirs),img_size],dtype=float)
    i=0
    # This would iterate through all the files
    for file in dirs:
        X[i]=plt.imread(path+file).flatten()
        i=i+1
    return X

def compute_mean(X):
    n=X.shape[0]
    d=X.shape[1]
    mu=np.empty([d,1],dtype=float)
    for i in range(0, d):
        mu[i][0]=0
        for j in range(0,n):
            mu[i][0]+=X[j][i]
        mu[i][0]=mu[i][0]/n
    return mu
def centre_data(X,mu):
    n=X.shape[0]
    d=X.shape[1]
    n_1= np.ones((n,1), dtype=float)
    X_ = X - np.matmul(n_1,mu.T)
    return X_

def reconstruction_error(X, X_prime):
    error = 0
    for i in range(rows-1):
        error+= np.linalg.norm(X[i]-X_prime[i])
    return error

def displayimage(data,data_recon,imgshape):
    """ display image"""
    #sort the
    fig = plt.figure()
    rands = [randint(0, rows-1) for p in range(samples)]
    print(rands)
    i=0
    for val in rands:
        #print(im.reshape(imgshape).shape)
        i=i+1
        img1 = data[val].reshape(imgshape)
        img2 = data_recon[val].reshape(imgshape)
        fig.add_subplot(2,2,i)
        plt.imshow(img1, cmap='gray')
        i=i+1
        fig.add_subplot(2,2,i)
        plt.imshow(img2, cmap='gray')
    plt.show()

img_size=77760 #243*320
shape = np.array([243,320])
path = "./yalefaces/"

X=load_images(path)
#print(X.shape)
mu = compute_mean(X)
print("mu shape:")
print(mu.shape)
X_=centre_data(X,mu)
print("X_ size:")
print(X_.shape[0])
print(X_.shape[1])
num_eigenvalues = 150
u, s, vt = sla.svds(X_, k=num_eigenvalues, ncv=None, tol=0, which='LM')
print("VT size:")
print(vt.shape)
Vp=vt.T
print("Vp size:")
print(Vp.shape)
Z=np.matmul(X_,Vp)
print("Z size:")
print(Z.shape)
n=X.shape[0]  
n_1= np.ones((n,1), dtype=float)
X_prime = np.matmul(n_1,mu.T) + np.matmul(Z,Vp.T)
print("X_prime size:")
print(X_prime.shape)


err=reconstruction_error(X, X_prime)
print("Error :"+str(err))


displayimage(X,X_prime,shape)