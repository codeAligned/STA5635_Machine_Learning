import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

class tispLogisticRegression:
    def __init__(self, xTr, yTr, xVal, yVal, case, maxIt=100):
        self.xTr=xTr
        self.yTr=yTr
        self.xVal=xVal
        self.yVal=yVal
        print('finish loading data')
        self.preProcess()
        self.case=case

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
        if self.case==1:
            #gisette
            lams=[0.187, 0.134, 0.0875, 0.053]
        elif self.case==2:
            # dexter
            lams=[0.141, 0.098, 0.0712, 0.0523]
        elif self.case==3:
            # madelon
            lams=[0.029795977, 0.0245, 0.01775, 0.0075]
        misclassTr=np.zeros(len(lams))
        misclassVal=np.zeros(len(lams))
        features=np.zeros(len(lams))
        for j in range(len(lams)):
            self.lam=lams[j]

            for i in range(self.maxIt):
                self.update()
                print(i, 'numbers of non zero weights: ', np.sum(self.w!=0.))

            wx=np.sum(self.xTr*self.w, axis=1)
            pred=np.ones(self.yTr.shape[0])
            pred[wx<0.]=-1.
            misclassTr[j]=1.-np.mean(pred==self.yTr)
            features[j]=np.sum(self.w!=0.)

            wx=np.sum(self.xVal*self.w, axis=1)
            pred=np.ones(self.yVal.shape[0])
            pred[wx<0.]=-1.
            misclassVal[j]=1.-np.mean(pred==self.yVal)
            print('lam: ', self.lam,', miss classification err in train, val: ',
                   misclassTr[j], misclassVal[j])
            self.w=np.zeros_like(self.w)
        return misclassTr, misclassVal, lams, features 


def project3a():
    xTr=np.loadtxt('gisette_train.data')
    yTr=np.loadtxt('gisette_train.labels')
    xVal=np.loadtxt('gisette_valid.data')
    yVal=np.loadtxt('gisette_valid.labels')
    lr=tispLogisticRegression(xTr, yTr, xVal, yVal, case=1)
    misclassTr, misclassVal, lam, features=lr.train()

    figureIndex=0
    plt.figure(figureIndex)
    plt.plot(features, misclassTr, 'o-', label='train')
    plt.plot(features, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.15])
    plt.legend()
    plt.show()

    myfile=open('gisette.dat', 'w')
    myfile.write('features lam misclassTr missclassVal\n')
    for i in range(features.shape[0]):
        myfile.write(str(features[i])+' '+str(lam[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()

def project3b():
    xTr=genfromtxt('dexter_train.csv', delimiter=',')
    xVal=genfromtxt('dexter_valid.csv', delimiter=',')
    yTr=np.loadtxt('dexter_train.labels' )
    yVal=np.loadtxt('dexter_valid.labels')
    lr=tispLogisticRegression(xTr, yTr, xVal, yVal, case=2)
    misclassTr, misclassVal, lam, features=lr.train()

    figureIndex=0
    plt.figure(figureIndex)
    plt.plot(features, misclassTr, 'o-', label='train')
    plt.plot(features, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.2])
    plt.legend()
    plt.show()

    myfile=open('dexter.dat', 'w')
    myfile.write('features lam misclassTr missclassVal\n')
    for i in range(features.shape[0]):
        myfile.write(str(features[i])+' '+str(lam[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()

def project3c():
    xTr=np.loadtxt('madelon_train.data')
    yTr=np.loadtxt('madelon_train.labels')
    xVal=np.loadtxt('madelon_valid.data')
    yVal=np.loadtxt('madelon_valid.labels')
    lr=tispLogisticRegression(xTr, yTr, xVal, yVal, case=3)
    misclassTr, misclassVal, lam, features=lr.train()

    figureIndex=0
    plt.figure(figureIndex)
    plt.plot(features, misclassTr, 'o-', label='train')
    plt.plot(features, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()

    myfile=open('madelon.dat', 'w')
    myfile.write('features lam misclassTr missclassVal\n')
    for i in range(features.shape[0]):
        myfile.write(str(features[i])+' '+str(lam[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()


project3a()
#project3b()
#project3c()
    
    




    
