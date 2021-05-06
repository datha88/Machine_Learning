import numpy as np
#from random import randint
import random
from matplotlib import pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z, deriv=False):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward(x, w0, w1):
    z1 = np.matmul(w0,x)
    x1 = sigmoid(z1)
    out = np.matmul( w1,x1)
    return out.T
def gradientdescent(w0,w1,gradient_dw0, gradient_dw1):
    step = 0.05
    w0 = w0 - step * gradient_dw0
    w1 = w1 - step * gradient_dw1
    return w0, w1

def backward(y, x, w0, w1,f):
    hingeloss = np.zeros(len(y))
    delta_l2 = np.zeros([len(y),1])

    ones = np.ones([len(y),1])
    hinge = ones - np.multiply(y,f)
    hinge[hinge < 0] = 0

    for i in range(len(y)):
        tmp = 1 - y[i] * f[i]
        hingeloss[i] = max(0, tmp)
        if tmp > 0:
            delta_l2[i] = -y[i]
    hingeloss = hinge

    z1 = np.matmul(w0,x)
    x1 = sigmoid(z1)
    diff_w1 = np.matmul( delta_l2.T,x1.T)
    
    delta_l1 = np.matmul(w1.T,delta_l2.T) * derivative_sigmoid(z1)
    diff_w0 = np.matmul(delta_l1,x.T)
    return diff_w0, diff_w1, hingeloss
def plot3d(data, estimate):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    # zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)
    Z = sigmoid(estimate)
    ax.scatter(data[:, 1], data[:, 2], Z)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f')
    plt.show()


# load the data
data = np.loadtxt("data2Class_adjusted.txt")


hL=100 
h0=3 
w0 = np.asarray([[random.uniform(-1, 1) for i in range(h0)] for j in range(hL)])
print("W0: ",w0.shape)
w1 = np.asarray([random.uniform(-1, 1) for i in range(hL)])
w1 = w1.reshape(1,hL)
print("W1:", w1.shape)
x = data[:, :3].reshape(3,len(data))

y = data[:, 3].reshape(len(data),1)
f = forward(x, w0, w1)
dw0, dw1, l = backward(y, x, w0, w1, f)
print(np.mean(l))
#i = 0
#err = 10
#while err>0.05:
#    f = forward(x, w0, w1)
#    dw0, dw1, l = backward(y, x, w0, w1, f)
#    w0, w1 = gradientdescent(w0, w1, dw0, dw1)
#    err = np.mean(l ** 2)
#    if i % 100 == 0:
#        print("Iter", i)
#        print("Error", err)
#    i += 1
#test = [1, 1.5, 1.4]
#pred = forward(test, w0, w1)
#pred[pred < 0] = -1
#pred[pred >= 0] = 1
#print(pred)
#plot3d(data, f)

    