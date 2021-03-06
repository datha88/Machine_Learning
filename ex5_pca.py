import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
from scipy import ndimage
import os
import numpy as np
from random import randint

dir = 'yalefaces'
neigenvalues = 150
rows = 166
columns = 77760
samples = 2

def readimages(folder):
    """
    #Read image data
    X = np.empty([166,77760])
    c=0
    for filename in os.listdir('yalefaces'):
        if not filename.endswith('.txt'):
            filename = 'yalefaces/'+filename
            im = ndimage.imread(filename)
            X[c]=im.reshape(77760)
            c = c + 1 
    """
    im = [ndimage.imread(folder+'/'+filename) for filename in os.listdir(folder)]
    X = np.asarray(im).reshape(rows,columns)
    imgshape = np.array(plt.imread(folder + '/' + 'subject01.gif')).shape
    print("Image shape:"+str(imgshape))
    return X, imgshape

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

data, shape = readimages(dir)

#sp.sparse.linalg.svds(data,k=neigenvalues)
mean = data.mean(0)
print(data.shape)
one = np.ones([rows])

Xt = data - one.reshape(rows,1) * mean.reshape(1,columns)
#covar = np.matmul(Xt.T, Xt)
u,s,v = sla.svds(Xt,neigenvalues,ncv=None, tol=0, which='LM')
#print(u.shape)
#print(s)
print(v.shape)

z = np.matmul(Xt, v.T)
#print(z.shape)

#reconstruction
data_recon = np.matmul(one.reshape(rows,1), mean.reshape(1,columns)) + np.matmul(z, v)

#error
error = data - data_recon
#print(np.matmul(error[0],error[0].T))

error = np.linalg.norm(error,axis=1)
#print(errnorm)
print("Error:",sum(error))

def cal_error(data,data_recon,err=0):
    finalerr = []
    e = np.empty([columns])
    print(e.shape)
    for i in range(rows):
        e[i] = np.asarray(data[i] - data_recon[i])
        err = np.linalg.norm(e[i])
        err = err * err
        finalerr.append(err)
    return finalerr


#finalerror = cal_error(data,data_recon)
#print(finalerror)
displayimage(data,data_recon,shape)

#print(data[0])
#print(data_recon[0])