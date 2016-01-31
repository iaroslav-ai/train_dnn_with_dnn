from ffnndatasets import FFNN, LinearModel, uniformInputs, modelDataset
from sklearn.svm import SVR

#model = FFNN(1,1,1)
model = LinearModel(5,1)
datasize = 2 ** 12;
samplesize = 16;

def randomInputs(shape):
    return uniformInputs(shape, samplesize)

X,Y = modelDataset(model, randomInputs, datasize)
Xv,Yv = modelDataset(model, randomInputs, datasize)
Xt,Yt = modelDataset(model, randomInputs, datasize)

Y = Y[:,0]
Yv = Yv[:,0]
Yt = Yt[:,0]


C = [0.1,1,10,100]
E = [0.01, 0.05, 0.1, 0.2]
bestsc = -1;

for c in C:
    for e in E:
        svr = SVR(C=c, epsilon=e)
        svr.fit(X, Y) 
        sc = svr.score(Xv, Yv)
        print sc
        if  sc > bestsc:
            sc = bestsc;
            be = e;
            bc = c;

svr = SVR(C=c, epsilon=e)
svr.fit(X, Y)    
sc = svr.score(Xt, Yt)
print "result:"
print sc      
