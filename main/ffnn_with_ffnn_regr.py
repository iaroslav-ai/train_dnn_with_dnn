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
    X = np.zeros((Size, (Nfeat + 1)*Npoints))
    Y = np.zeros((Size, (Nfeat + 2)*Nneurons))
    
    for i in range(Size):
        # generate random ffnn
        P = FFNN_Parameters(Nfeat, Nneurons, 1, 20)
        x = rnd(Npoints, Nfeat)*20
        y = np.dot(x, P[0]) + P[1]
        xy = np.concatenate((x,y), axis=1)
        
        W = np.vstack((P[0],[P[1]] )) #, np.transpose( P[2] )
        W = np.transpose(W)
        W = W[np.lexsort(np.fliplr(W).T)]
        W = np.transpose(W)
        
        xval = xy.flatten()
        yval = P[0].flatten()
        
        X[i,:] = xval
        Y[i,:] = yval
    
    return X,Y

def dataset():
    return GenerateDataset(10000, 1, 8, 1)

X,Y = dataset()
Xt,Yt = dataset()

def objective(X,Y,P):
    Yp = FFNN(X, P)
    diff = Y - Yp
    obj = np.mean( diff**2 )
    return obj

def train(P):
    return objective(X, Y, P)

def test(P):
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

for i in range(10000):
    G = grd(P)
    P, M, Gsq = adam(P, G, M, Gsq, alpha=0.001)
    tst = test(P)
    print  tst
    