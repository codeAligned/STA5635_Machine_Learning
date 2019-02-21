import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self, file_xTr, file_yTr, file_xVal, file_yVal,
            maxIt=1000, eta=0.01):
        self.xTr=np.loadtxt(file_xTr)
        self.yTr=np.loadtxt(file_yTr)
        self.xVal=np.loadtxt(file_xVal)
        self.yVal=np.loadtxt(file_yVal)
        print('finish loading data')
        self.preProcess()

        self.w=np.zeros(self.xTr.shape[1])
        self.lam=0.001
        self.eta=eta
        self.maxIt=maxIt

    def preProcess(self):
        xStd=np.std(self.xTr, axis=0)
        mask=(xStd!=0.)
        self.xTr=self.xTr[:, mask]
        meanX=np.mean(self.xTr, axis=0)
        stdX=np.std(self.xTr, axis=0)
        self.xTr=(self.xTr-meanX)/stdX
        self.xVal=self.xVal[:, mask]
        self.xVal=(self.xVal-meanX)/stdX

        self.xTr=np.insert(self.xTr, 0, 1., axis=1)
        self.xVal=np.insert(self.xVal, 0, 1., axis=1)

        self.yTr[self.yTr==-1.]=0.
        self.yVal[self.yVal==-1.]=0.
        print('finish preprocessing data')

    def gradient(self):
        wx=np.sum(self.xTr*self.w, axis=1)
        exp_wx=np.exp(wx)
        print
        grad=np.sum((self.yTr-exp_wx/(1.+exp_wx))*(self.xTr).T, axis=1)
        return grad

    def update(self):
        grad=self.gradient()
        self.w=(1.-self.eta*self.lam)*self.w+self.eta/self.xTr.shape[0]*grad

    def train(self):
        loglike=np.zeros(self.maxIt)
        misclassTr=np.zeros(self.maxIt)
        misclassVal=np.zeros(self.maxIt)
        for i in range(self.maxIt):
            self.update()
            wx=np.sum(self.xTr*self.w, axis=1)
            loglike[i]=np.sum(self.yTr*wx-np.log(1.+np.exp(wx)), axis=0)
            pred=((wx>=0.)==self.yTr)
            misclassTr[i]=1.-np.mean(pred)

            wx=np.sum(self.xVal*self.w, axis=1)
            pred=((wx>=0.)==self.yVal)
            #if i%10==0:
            #    for j in range(10):
            #        print(wx[j], self.yVal[j])
            misclassVal[i]=1.-np.mean(pred)
            print(i, 'loglike, miss classification err in train, val: ',
                    loglike[i], misclassTr[i], misclassVal[i])
        return loglike, misclassTr, misclassVal


def project3a():
    lr=logisticRegression('gisette_train.data', 'gisette_train.labels',
                           'gisette_valid.data', 'gisette_valid.labels')
    loglike, misclassTr, misclassVal=lr.train()

    myfile=open('gisette.dat', 'w')
    for i in range(loglike.shape[0]):
        myfile.write(str(loglike[i])+' '+str(misclassTr[i])+
                                     ' '+str(misclassVal[i])+'\n')
    myfile.close()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), loglike, label='log-likelihood')
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')

    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), misclassTr, label='train')
    plt.plot(np.arange(loglike.shape[0]), misclassVal, label='test')
    plt.xlabel('iteration')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.3])
    plt.legend()
    plt.show()

def project3b():
    lr=logisticRegression('madelon_train.data', 'madelon_train.labels',
                          'madelon_valid.data', 'madelon_valid.labels',
                          eta=0.025)
    loglike, misclassTr, misclassVal=lr.train()

    myfile=open('madelon.dat', 'w')
    for i in range(loglike.shape[0]):
        myfile.write(str(loglike[i])+' '+str(misclassTr[i])+
                                     ' '+str(misclassVal[i])+'\n')
    myfile.close()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), loglike, label='log-likelihood')
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')

    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), misclassTr, label='train')
    plt.plot(np.arange(loglike.shape[0]), misclassVal, label='test')
    plt.xlabel('iteration')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.6])
    plt.legend()
    plt.show()

def project3c():
    lr=logisticRegression('X.dat', 'Y.dat','Xtest.dat', 'Ytest.dat',
                          maxIt=20000, eta=0.05)
    loglike, misclassTr, misclassVal=lr.train()

    myfile=open('hill_valley.dat', 'w')
    for i in range(loglike.shape[0]):
        myfile.write(str(loglike[i])+' '+str(misclassTr[i])+
                                     ' '+str(misclassVal[i])+'\n')
    myfile.close()

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), loglike, label='log-likelihood')
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')

    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(np.arange(loglike.shape[0]), misclassTr, label='train')
    plt.plot(np.arange(loglike.shape[0]), misclassVal, label='test')
    plt.xlabel('iteration')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.6])
    plt.legend()
    plt.show()

project3a()
#project3b()
#project3c()
    
    




    
