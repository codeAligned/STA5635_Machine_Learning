import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

class NN:
    def __init__(self, xTr, yTr, xVal, yVal,
            nIter=20, eta=0.0025, k=32, layers=2,
            inputs=50):
        self.xTr=xTr
        self.yTr=yTr
        self.xVal=xVal
        self.yVal=yVal
        self.nIter=nIter
        self.k=k
        self.lr=eta
        self.layers=layers
        self.inputs=inputs

        self.preProcess()
        # M is based on the duplicate features removed version
        self.M=self.xTr.shape[1]

    def preProcess(self):
        xStd=np.std(self.xTr, axis=0)
        mask=(xStd!=0.)
        self.xTr=self.xTr[:, mask]
        meanX=np.mean(self.xTr, axis=0)
        stdX=np.std(self.xTr, axis=0)
        self.xTr=(self.xTr-meanX)/stdX
        self.xVal=self.xVal[:, mask]
        self.xVal=(self.xVal-meanX)/stdX

        #self.xTr=np.insert(self.xTr, 0, 1., axis=1)
        #self.xVal=np.insert(self.xVal, 0, 1., axis=1)
        #self.yTr[self.yTr==0.]=-1.
        #self.yVal[self.yVal==0.]=-1.
        print('finish preprocessing data')

    def train(self):
        model = Sequential()
        model.add(Dense(self.k, input_dim=self.inputs, activation='relu'))
        if self.layers==3:
            model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        optimizer=keras.optimizers.RMSprop(lr=self.lr, rho=0.9, 
                                     epsilon=None, decay=0.0)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit(self.xTr, self.yTr,
                  epochs=self.nIter,
                  batch_size=128)
        scoreTr = model.evaluate(self.xTr, self.yTr, batch_size=128)
        scoreVal = model.evaluate(self.xVal, self.yVal, batch_size=128)

        print('miss classification err in train, val: ',
                1.-scoreTr[1], 1.-scoreVal[1])
        return  1.-scoreTr[1], 1.-scoreVal[1]

def project9a():
    # 36499 93565
    x=np.loadtxt('MiniBooNE_PID.txt')
    y=np.zeros((x.shape[0],1))
    y[36499:,0]=1
    x0=np.append(x, y, axis=1)

    ks=[32, 64, 128, 256]
    alphas=[0.01, 0.01, 0.01, 0.01]
    m=x0.shape[0]
    runs=10
    misclassTr=np.zeros((len(ks), runs))
    misclassVal=np.zeros((len(ks), runs))
    for k in range(len(ks)):
        for run in range(runs):
            x=np.copy(x0)
            np.random.shuffle(x)
            y=x[:,-1]
            x=x[:,:-1]
            x_train=x[:int(0.8*m)]
            y_train=y[:int(0.8*m)]
            x_test=x[int(0.8*m):]
            y_test=y[int(0.8*m):]
            model=NN(x_train, y_train, x_test, y_test, k=ks[k],
                    eta=alphas[k])
            misclassTr[k, run], misclassVal[k, run]=model.train()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(len(ks)), np.mean(misclassTr, axis=1), label='train')
    plt.plot(np.arange(len(ks)), np.mean(misclassVal, axis=1), label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')

    myfile=open('miniboone_a.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(np.mean(misclassTr[i]))+
                                ' '+str(np.mean(misclassVal[i]))+'\n')
    myfile.close()
    plt.show()

def project9b():
    # 36499 93565
    x=np.loadtxt('MiniBooNE_PID.txt')
    y=np.zeros((x.shape[0],1))
    y[36499:,0]=1
    x0=np.append(x, y, axis=1)

    ks=[32, 64, 128, 256]
    alphas=[0.01, 0.01, 0.01, 0.01]
    m=x0.shape[0]
    runs=10
    misclassTr=np.zeros((len(ks), runs))
    misclassVal=np.zeros((len(ks), runs))
    for k in range(len(ks)):
        for run in range(runs):
            x=np.copy(x0)
            np.random.shuffle(x)
            y=x[:,-1]
            x=x[:,:-1]
            x_train=x[:int(0.8*m)]
            y_train=y[:int(0.8*m)]
            x_test=x[int(0.8*m):]
            y_test=y[int(0.8*m):]
            model=NN(x_train, y_train, x_test, y_test, k=ks[k],
                    eta=alphas[k], layers=3)
            misclassTr[k, run], misclassVal[k, run]=model.train()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(len(ks)), np.mean(misclassTr, axis=1), label='train')
    plt.plot(np.arange(len(ks)), np.mean(misclassVal, axis=1), label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')

    myfile=open('miniboone_b.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(np.mean(misclassTr[i]))+
                                ' '+str(np.mean(misclassVal[i]))+'\n')
    myfile.close()
    plt.show()

def project9c():
    xTr=np.loadtxt('madelon_train.data')
    yTr=np.loadtxt('madelon_train.labels')
    yTr[yTr==-1]=0
    xVal=np.loadtxt('madelon_valid.data')
    yVal=np.loadtxt('madelon_valid.labels')
    yVal[yVal==-1]=0

    ks=[32, 64, 128, 256]
    alphas=[0.01, 0.01, 0.01, 0.01]
    runs=10
    misclassTr=np.zeros((len(ks), runs))
    misclassVal=np.zeros((len(ks), runs))
    for k in range(len(ks)):
        for run in range(runs):
            model=NN(np.copy(xTr), np.copy(yTr),
                     np.copy(xVal), np.copy(yVal), k=ks[k],
                     eta=alphas[k], nIter=100, inputs=500)
            misclassTr[k, run], misclassVal[k, run]=model.train()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(len(ks)), np.mean(misclassTr, axis=1), label='train')
    plt.plot(np.arange(len(ks)), np.mean(misclassVal, axis=1), label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')

    myfile=open('madelon_a.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(np.mean(misclassTr[i]))+
                                ' '+str(np.mean(misclassVal[i]))+'\n')
    myfile.close()
    plt.show()

def project9d():
    xTr=np.loadtxt('madelon_train.data')
    yTr=np.loadtxt('madelon_train.labels')
    yTr[yTr==-1]=0
    xVal=np.loadtxt('madelon_valid.data')
    yVal=np.loadtxt('madelon_valid.labels')
    yVal[yVal==-1]=0

    ks=[32, 64, 128, 256]
    alphas=[0.001, 0.001, 0.001, 0.001]
    runs=10
    misclassTr=np.zeros((len(ks), runs))
    misclassVal=np.zeros((len(ks), runs))
    for k in range(len(ks)):
        for run in range(runs):
            model=NN(np.copy(xTr), np.copy(yTr),
                     np.copy(xVal), np.copy(yVal), k=ks[k],
                     eta=alphas[k], nIter=20, inputs=500,
                     layers=3)
            misclassTr[k, run], misclassVal[k, run]=model.train()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(len(ks)), np.mean(misclassTr, axis=1), label='train')
    plt.plot(np.arange(len(ks)), np.mean(misclassVal, axis=1), label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')

    myfile=open('madelon_b.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(np.mean(misclassTr[i]))+
                                ' '+str(np.mean(misclassVal[i]))+'\n')
    myfile.close()
    plt.show()

#project9a()
#project9b()
#project9c()
project9d()
    
    




    
