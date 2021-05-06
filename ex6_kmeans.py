import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os

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

path = "./yalefaces_cropBackground/"
img_size = 38880 #(243, 160, 0)
K=[2,3,4,5,6,7,8]
X, N=load_images(path)
J=[]
for k in K:
    best_mu = np.zeros((k, X.shape[1]))
    r ,best_mu, min_J= k_mean(X, N, k)
    J.append(min_J)
plt.plot(K,J)
plt.show()

# part a
k=4
best_mu = np.zeros((k, X.shape[1]))
r ,best_mu, min_J= k_mean(X, N, k)

shape = np.array([243,160])
for i in best_mu:
    imgplot = i.reshape(shape)
    plt.figure() 
    plt.imshow(imgplot, cmap='gray')
plt.show()