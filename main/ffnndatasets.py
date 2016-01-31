'''
contains classes and methods for evaluation of the prediction of ffnn weights
'''

import numpy as np

class Model():
    def paramsToVector(self):
        flat = []
        for p in self.P:
            flat.append(p.flatten())
        return np.concatenate(flat);
    def transferShape(self):
        return self.isz, self.osz;

class FFNN(Model):
    def __init__(self, isz, hsz, osz):
        self.isz = isz;
        self.hsz = hsz;
        self.osz = osz;
        self.randomize();
        
    def randomize(self):
        P = [];
        # projection matrix
        P.append(np.random.rand(self.isz, self.hsz))
        # bias of neurons
        P.append(np.random.rand(self.hsz))
        # combination values
        P.append(np.random.rand(self.hsz, self.osz))
        self.P = P;
    
    def compute(self,X):
        H = np.dot( X, self.P[0] ) + self.P[1];
        H = np.maximum(H, H*0.02)
        return np.dot( H , self.P[2] )

class LinearModel(Model):
    def __init__(self, isz, osz):
        self.isz = isz;
        self.osz = osz;
        self.randomize();
        
    def randomize(self):
        P = [];
        # combination values
        P.append(np.random.rand(self.isz, self.osz))
        self.P = P;
    
    def compute(self,X):
        return np.dot( X , self.P[0] )


def uniformInputs(shape, samples):
    return np.random.rand(samples, shape[0])

def modelDataset(model, inputs, size):
    
    X = [];
    Y = [];
    
    for i in range(size):
        model.randomize();
        xx = inputs(model.transferShape());
        xy = model.compute(xx);
        x = np.concatenate((xx, xy), axis=1);
        x = x.flatten();
        y = model.paramsToVector();
        
        if len(X) == 0:
            X = np.zeros((size, x.shape[0]))
            Y = np.zeros((size, y.shape[0]))
        
        X[i,:] = x;
        Y[i,:] = y;
    
    return X, Y
    
    
    
    
    
    
    
    