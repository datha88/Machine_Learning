from operator import inv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functools import reduce

def plotData(data):
    X = data[:, :2]
    x1 = []
    x2 = []
    for point in data:
        if point[2] == 1:
            x1.append(point[0])
            x2.append(point[1])
    plt.plot(x1,x2,'ro')

    x1 = []
    x2 = []
    for point in data:
        if point[2] == 0:
            x1.append(point[0])
            x2.append(point[1])
    plt.plot(x1,x2,'go')

def prepend_one(X):
	"""prepend a one vector to X."""
	return np.column_stack([np.ones(X.shape[0]), X])

def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!"""
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])

def linear(data):
    X= data[:, :2]
    X = prepend_one(X)
    return X
def quadratic(data):
    X= data[:, :2]
    X = prepend_one(X)
    X = np.column_stack([X, data[:, 0] * data[:, 0], data[:, 0] * data[:, 1], data[:, 1] * data[:, 1]])
    return X
def sigmoid(z):
    return np.exp(z) / (1+ np.exp(z))#np.array(1/(1 + np.exp(-z)))
    # sigmoid size n x 1
def gradient(X,y,beta,lamda):
    I = np.identity(X.shape[1])
    p = sigmoid(np.matmul(X, beta))
    return np.matmul(X.T, (p-y)) + 2 * lamda * np.matmul(I,beta)
    #size k x 1

def hessian(X,lamda):
    p1 = sigmoid(np.matmul(X, beta))
    I = np.eye(X.shape[1])
    return mdot(X.T,diag(p1),X) + 2*lamda * I

def diag(z):
    return np.diagflat(np.multiply(z, 1-z))


def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def newtonmethod(X,y,beta,lamda,step):
    beta[:] = 0
    for i in range(step):
        H = hessian(X, lamda)
        G = gradient(X, y, beta, lamda)
        delta = np.dot(np.linalg.inv(H), G)
        beta = beta - np.dot(np.linalg.inv(H), G)
    return beta



def mean_neg_log_likelihood(X,beta):
    return (-1 * np.log(sigmoid(np.matmul(X,beta))) /200)

def plot_prob(X,y,opti,linflag,probflag):
    X_grid = prepend_one(grid2d(-3, 3, num=30))
    if linflag == 2:
        X_grid = np.column_stack([X_grid, X_grid[:, 1] * X_grid[:, 1], X_grid[:, 1] * X_grid[:, 2], X_grid[:, 2] * X_grid[:, 2]])

    if probflag == 1:
        y_grid = sigmoid(np.matmul(X_grid, opti))
    else:
        y_grid = mdot(X_grid, opti)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
    ax.set_title("Proability y=1 over test grid")
    plt.show()

lamda = 0.1**5;
print("Lamda",lamda)
data = np.loadtxt("data2Class.txt")
X = linear(data) #size n x k
#X = quadratic(data)
#plotData(data)
#plt.show()-

print("Size of X:",X.shape)
y = data[:, 2] #size n x 1
print("Size of y:",y.shape)


beta = np.empty(X.shape[1])


step = 10
op_beta = newtonmethod(X,y,beta,lamda,step)
print ("Optimim beta\n", op_beta)


mean_nll = mean_neg_log_likelihood(X,beta)
print("Mean neg log likelihood:",mean_nll.mean())
featureflag = 1
probflag = 1 
plot_prob(X,y,op_beta,featureflag,probflag)



