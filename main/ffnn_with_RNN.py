import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import time

# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

srng = RandomStreams(seed=255)

def rnd(shape):
    return (srng.uniform(shape)*2 - 1)

def predict_ffnn(X, W, b, s):

    y = T.zeros([datasz, batchsz, 1])

    for i in range(batchsz):
        # generate their activations
        H = T.nnet.relu( T.dot(X[:,i,:],W[i,:]) + b[i,:] , 0.01)
        v = T.dot(H, s[i,:])
        y = T.set_subtensor(y[:,i,0], v)
    
    return y 

def rnd_sh(*args, **kwargs):
    return theano.shared( np.random.randn(*args, **kwargs)*0.05 )

def RNN_Parameters(inputSize, hiddenLayer, outputSize):
    P = []
    # W
    P.append( rnd_sh(inputSize, hiddenLayer) ) 
    P.append( rnd_sh(hiddenLayer, hiddenLayer) )
    # Bias
    P.append( rnd_sh(hiddenLayer) ) 
    # linear output
    P.append( rnd_sh(hiddenLayer, outputSize) ) 
    
    return P

def RNN_Forward(x,H,P):
    H = T.dot(x, P[0]) + T.dot(H, P[1]) + P[2];
    H = T.nnet.relu(H, 0.01);
    y = T.dot(H, P[-1])
    return y, H

def RNN_Init(X,P):
    return T.dot(X[0,], P[0])*0;
  
def RNN(X, H, P):
    
    for i in range(datasz):
        yp, H = RNN_Forward(X[i,:], H, P)
    
    return yp, H

batchsz = 16
datasz = 64
neurons_generator = 5;
features = 1
rnn_size = 128

W = rnd([batchsz, features, neurons_generator])
b = rnd([batchsz, neurons_generator])
s = rnd([batchsz, neurons_generator])

# generate batchsz random ffnn's
X = rnd([datasz, batchsz,  features])
y = predict_ffnn(X, W, b, s)

Xt = rnd([datasz, batchsz, features])
yt = predict_ffnn(Xt, W, b, s)

P = RNN_Parameters(features + 1, rnn_size, (features+2)*neurons_generator)

obj = T.zeros(())

# input for the network: features X and outputs y concatenated
Input = T.concatenate((X,y), axis=2)
# initialize initial state of the RNN
H = RNN_Init(Input, P)

reps = 1
for i in range(reps):
    # Pass RNN through the dataset
    Pr, H = RNN(Input, H, P)
    
    # reshape predicted parameters into network
    Pr = T.reshape(Pr, (batchsz, features+2, neurons_generator), ndim = 3 )
    Wt = Pr[:, :-2,:]
    bt = Pr[:, -2,:]
    st = Pr[:, -1,:]
    
    # compute generalization of the network
    yp = predict_ffnn(Xt, Wt, bt, st)
    obj = obj + T.mean( abs( yt - yp ) )/reps;

grd = T.grad(obj, P)

upd = []

for i in range(len(grd)):
    tpl = (P[i], P[i] - 0.0001*grd[i])
    upd.append(tpl)

print "start compilation ... "
start_time = time.time()
r = function( inputs = [], outputs = obj, updates=upd )
comp_time = time.time()-start_time
print "compiled in", comp_time, "sec"


runavg = r()
mx = runavg;
mn = runavg;

a = 0.95;
for i in range(10000):
    runavg = a*runavg + (1-a)*r()
    mx = max(mx, runavg)
    mn = min(mn, runavg)
    print mn / mx, runavg / mx