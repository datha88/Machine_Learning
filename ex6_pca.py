import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os
import scipy.sparse.linalg as sla
from random import randint

def load_images(path):
    dirs = os.listdir( path )
    X = np.empty([len(dirs),img_size],dtype=float)
    i=0
    # This would iterate through all the files
    for file in dirs:
        im = plt.imread(path+file)
        im2 = im[:, :, 0]
        X[i]=im2.flatten()
        i=i+1
    return X, i

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

path = "./yalefaces_cropBackground/"
img_size = 38880
rows = 136
columns = 38880
samples = 2
shape = np.array([243,160])
X,N = load_images(path)
print(X.shape)

mu = compute_mean(X)
print("mu shape:")
print(mu.shape)
X_=centre_data(X,mu)
print("X_ size:")
print(X_.shape[0])
print(X_.shape[1])
num_eigenvalues = 20
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

def closest_cluster_center(X, N, k, mu):
    r = np.zeros((N, k))
    for n in range(0,N):
        distanceArray= []
        for i in range(0, k):
            dist = np.linalg.norm(X[n] - mu[i]) ** 2
            distanceArray.append(dist)
        mindistIndex = np.argmin(distanceArray)
        r[n][mindistIndex] = 1
    return r
def compute_J(X,r,mu,k,N):
    for i in range(0,N):
        J=0
        for j in range(0,k):
            norm = np.linalg.norm(X[i] - mu[j]) ** 2
            J+= norm*r[i][j]
    return J
def different_mu(X, N, k, r):
    new_mu = np.zeros((k, X.shape[1]))
    for i in range(0, k):
        sum = 0
        count = 0
        for j in range(0, N):
            if r[j][i] == 1:
                count+=1
                sum += X[j]
        if(count ==0):
            count=1
        center = sum/count
        new_mu[i]=center
    return new_mu

def k_mean(X, N, k):
    mu = np.random.uniform(0, 255, (k, X.shape[1]))
    r=closest_cluster_center(X, N, k, mu)
    J=compute_J(X,r,mu,k,N)
    min_J = J
    for i in range(0,10):
        mu = different_mu(X, N, k, r)
        r = closest_cluster_center(X,N, k, mu)
        new_J =  compute_J(X , r, mu, k, N)
        if new_J < min_J:
            best_mu = mu
            min_J = new_J        
    return r, np.asarray(best_mu),min_J
k=4
best_mu = np.zeros((k, X_prime.shape[1]))
r ,best_mu, min_J= k_mean(X_prime, N, k)

shape = np.array([243,160])
for i in best_mu:
    imgplot = i.reshape(shape)
    plt.figure() 
    plt.imshow(imgplot, cmap='gray')
plt.show()