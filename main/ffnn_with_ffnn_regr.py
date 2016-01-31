'''
Created on Jan 8, 2016

@author: Iaroslav
'''
from autograd import grad
import autograd.numpy as np
import copy

def rnd(*args, **kwargs):
    return np.random.randn(*args, **kwargs)*0.05

def FFNN_Parameters(input, hidden, output, scale=1.0):
    P = []
    # W paramter
    P.append( rnd(input, hidden)*scale )
    # bias
    P.append( rnd(hidden)*scale )
    # lin combination
    P.append( rnd(hidden, output)*scale )
    return P

def FFNN(X, P):
    H = np.dot(X,P[0]) + P[1]
    H = np.maximum(H, H*0.01)
    return np.dot(H, P[2])

def GenerateDataset(Size, Nfeat, Npoints, Nneurons):
    
    X, Y = [], []
    
    for i in range(Size):
        # generate random ffnn
        #P = FFNN_Parameters(Nfeat, Nneurons, 1, 20)
        x = rnd(Npoints, Nfeat)*20
        #P[2] = np.abs(P[2])
        #y = np.dot( np.tanh( np.dot(x, P[0]) + P[1]),  P[2] )
        pr = np.random.randn(x.shape[1], 1)
        pr = pr > 0
        y = np.dot(x, pr)*0.5
        xy = np.concatenate((x,y), axis=1)
        """
        W = np.vstack((P[0], [P[1]] , np.transpose( P[2] ))) #
        W = np.transpose(W)
        W = W[np.lexsort(np.fliplr(W).T)]
        W = np.transpose(W)"""
        
        xval = xy.flatten()
        yval = pr.flatten()
        
        if X == []:
            X = np.zeros((Size, xval.shape[0]))
            Y = np.zeros((Size, yval.shape[0]))
        
        X[i,:] = xval
        Y[i,:] = yval
    
    return X,Y

def dataset():
    return GenerateDataset(256, 10, 4, 2)

X,Y = dataset()
Xt,Yt = dataset()

def objective(X,Y,P):
    Yp = FFNN(X, P)
    mxAbs = np.max( np.abs(Y) )
    diff = (Y - Yp)
    obj = np.mean( np.abs( diff ) )
    return obj

def train(P):
    X,Y = dataset()
    return objective(X, Y, P)

def test(P):
    Xt,Yt = dataset()
    return objective(Xt, Yt, P)

def adam(P, G, M, Gsq, alpha):
    
    if M is None:
        M = copy.deepcopy(G)
    
    if Gsq is None:
        Gsq = []
        for i in range(len(G)):
            Gsq.append(G[i]*0+1);
    
    for i in range(len(G)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        Gsq[i] = 0.9 * Gsq[i] + 0.1 * M[i]*M[i]
        P[i] = P[i] - alpha* M[i] / np.sqrt( Gsq[i] + 1e-6 )

    return P, M, Gsq

P = FFNN_Parameters(X.shape[1], 128, Y.shape[1])

grd = grad(train)

M, Gsq = None, None
avg = 1;

for i in range(100000):
    G = grd(P)
    P, M, Gsq = adam(P, G, M, Gsq, alpha=0.001)
    tst = test(P)
    avg = avg*0.9 + 0.1*tst
    print  avg
    