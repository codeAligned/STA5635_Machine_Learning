import numpy as np
import matplotlib.pyplot as plt

class tispLogisticRegression:
    def __init__(self, file_xTr, file_yTr, file_xVal, file_yVal,
            maxIt=100):
        self.xTr=np.loadtxt(file_xTr)
        self.yTr=np.loadtxt(file_yTr)
        self.xVal=np.loadtxt(file_xVal)
        self.yVal=np.loadtxt(file_yVal)
        print('finish loading data')
        self.preProcess()

        self.w=np.zeros(self.xTr.shape[1])
        self.lam=0.001
        self.eta=1./self.xTr.shape[0]
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

        self.yTr[self.yTr==0.]=-1.
        self.yVal[self.yVal==0.]=-1.
        print('finish preprocessing data')

    def gradient(self):
        wx=np.sum(self.xTr*self.w, axis=1)
        temp=self.yTr/(1.+np.exp(self.yTr*wx))
        grad=np.sum(temp*(self.xTr).T, axis=1)
        return grad

    def update(self):
        grad=self.gradient()
        self.w+=grad*self.eta
        self.w[np.absolute(self.w)<=self.lam]=0.

    def train(self):
        target=np.array([10, 30, 100, 300])
        #lams=[0.029795977]
        #lams=[0.0245]
        #lams=[0.01775]
        lams=[0.005]
        #lams=[0.032, 0.021, 0.005, 0.001]
        for j in range(len(lams)):
            self.lam=lams[j]

            for i in range(self.maxIt):
                self.update()
                print(i, 'numbers of non zero weights: ', np.sum(self.w!=0.))

            wx=np.sum(self.xTr*self.w, axis=1)
            pred=np.ones(self.yTr.shape[0])
            pred[wx<0.]=-1.
            misclassTr=1.-np.mean(pred==self.yTr)

            wx=np.sum(self.xVal*self.w, axis=1)
            pred=np.ones(self.yVal.shape[0])
            pred[wx<0.]=-1.
            misclassVal=1.-np.mean(pred==self.yVal)
            print('lam: ', self.lam,', miss classification err in train, val: ',
                   misclassTr, misclassVal)
        return misclassTr, misclassVal, self.lam, np.sum(self.w!=0.)


def project3a():
    lr=tispLogisticRegression('gisette_train.data', 'gisette_train.labels',
                           'gisette_valid.data', 'gisette_valid.labels')
    misclassTr, misclassVal, lam, features=lr.train()

    #figureIndex=0
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(np.arange(loglike.shape[0]), loglike, label='log-likelihood')
    #plt.xlabel('iteration')
    #plt.ylabel('log-likelihood')

    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(np.arange(loglike.shape[0]), misclassTr, label='train')
    #plt.plot(np.arange(loglike.shape[0]), misclassVal, label='test')
    #plt.xlabel('iteration')
    #plt.ylabel('miss classification ratio')

    #plt.ylim([0, 0.3])
    #plt.legend()
    #plt.show()

def project3b():
    lr=tispLogisticRegression('madelon_train.data', 'madelon_train.labels',
                          'madelon_valid.data', 'madelon_valid.labels')
    misclassTr, misclassVal, lam, features=lr.train()

    #myfile=open('madelon.dat', 'w')
    #for i in range(loglike.shape[0]):
    #    myfile.write(str(loglike[i])+' '+str(misclassTr[i])+
    #                                 ' '+str(misclassVal[i])+'\n')
    #myfile.close()

    #figureIndex=0
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(np.arange(loglike.shape[0]), loglike, label='log-likelihood')
    #plt.xlabel('iteration')
    #plt.ylabel('log-likelihood')

    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(np.arange(loglike.shape[0]), misclassTr, label='train')
    #plt.plot(np.arange(loglike.shape[0]), misclassVal, label='test')
    #plt.xlabel('iteration')
    #plt.ylabel('miss classification ratio')

    #plt.ylim([0, 0.6])
    #plt.legend()
    #plt.show()

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

#project3a()
project3b()
#project3c()
    
    




    
