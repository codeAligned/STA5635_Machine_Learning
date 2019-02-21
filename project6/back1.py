import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import genfromtxt

class FSA:
    def __init__(self, xTr, yTr, xVal, yVal, n_feature, maxIt=500):
        self.xTr=xTr
        self.yTr=yTr
        self.xVal=xVal
        self.yVal=yVal
        self.k=n_feature
        print('finish loading data')
        
        self.preProcess()
        self.M=self.xTr.shape[1]
        self.w=np.zeros(self.M)
        self.eta=20
        self.maxIt=maxIt
        self.s=0.001
        self.mu=100
        self.mind=[]
        
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

        self.yTr[self.yTr==0.]=-1.
        self.yVal[self.yVal==0.]=-1.
        print('finish preprocessing data')

    def gradient(self):
        wx=self.xTr*self.w
        temp=np.sum((wx).T*self.yTr, axis=0)
        ind=(temp<=1)
        temp1=(temp[ind]-1)
        grad=np.sum(2*temp1*(self.yTr[ind]*(self.xTr[ind,:]).T)/(1+temp1**2), axis=1)
        return grad
# self.xTr can not be change by mind in order to avoid reload data every time
    def update(self,i):
        grad=self.gradient()
        self.w=grad*self.eta+(1-2*self.eta*self.s)*self.w
        Mi=self.k+(self.M-self.k)*np.max([0,(self.maxIt-2*i)/(2*i*self.mu+self.maxIt)])
        mask=np.argsort(np.abs(self.w))
        self.mind=mask[-int(Mi):]
        self.w=self.w[self.mind]
        self.xTr=self.xTr[:,self.mind]
        
    def Ld(self):
        wx=self.xTr*self.w
        temp=np.sum((wx).T*self.yTr, axis=0)
        ind=(temp<=1.0)
        temp1=(temp[ind]-1.0)
        L=np.sum(np.log(1+temp1**2.0))+self.s*LA.norm(self.w)
        return L
    
    def train(self):
        trainloss=np.zeros(self.maxIt)      
        for i in range(self.maxIt):
            self.update(i)
            trainloss[i]=self.Ld()
            
        if self.k==10:
            figureIndex=0
            plt.figure(figureIndex)
            plt.plot(range(self.maxIt), trainloss, '-', label='trainloss')
            plt.xlabel('num of iterations')
            plt.ylabel('train loss')

            #plt.ylim([0, 0.15])
            plt.legend()
            plt.show()  
        
        wx=np.sum(self.xTr*self.w, axis=1)
        pred=np.ones(self.yTr.shape[0])
        pred[wx<0.]=-1.
        misclassTr=1.-np.mean(pred==self.yTr)

        wx=np.sum(self.xVal[:,self.mind]*self.w, axis=1)
        pred=np.ones(self.yVal.shape[0])
        pred[wx<0.]=-1.
        misclassVal=1.-np.mean(pred==self.yVal)
        print('miss classification err in train, val: ',
                   misclassTr, misclassVal)
        self.w=np.zeros_like(self.w)
        return misclassTr, misclassVal


def project3a():
    
    k=np.array([10,30,100,300])
    nk=len(k)
    misclassTr=np.zeros(nk)
    misclassVal=np.zeros(nk)
    for j in range(nk):
        xTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\Gisette\gisette_train.txt')
        yTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\Gisette\gisette_trainlabels.txt')
        xVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\Gisette\gisette_valid.txt')
        yVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\Gisette\gisette_validlabels.txt')
        
        lr=FSA(xTr, yTr, xVal, yVal, k[j])
        misclassTr[j], misclassVal[j]=lr.train()

    figureIndex=1
    plt.figure(figureIndex)
    plt.plot(k, misclassTr, 'o-', label='train')
    plt.plot(k, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    #plt.ylim([0, 0.15])
    plt.legend()
    plt.show()

    myfile=open('gisette.dat', 'w')
    myfile.write('misclassTr missclassVal\n')
    for i in range(k.shape[0]):
        myfile.write(str(k[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()

def project3b():
    k=np.array([10,30,100,300])
    nk=len(k)
    misclassTr=np.zeros(nk)
    misclassVal=np.zeros(nk)
    for j in range(nk):

        xTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\dexter\dexter_train.txt')
        yTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\dexter\dexter_trainlabels.txt')
        xVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\dexter\dexter_valid.txt')
        yVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\dexter\dexter_validlabels.txt')
        
        lr=FSA(xTr, yTr, xVal, yVal, k[j])
        misclassTr[j], misclassVal[j]=lr.train()

    figureIndex=1
    plt.figure(figureIndex)
    plt.plot(k, misclassTr, 'o-', label='train')
    plt.plot(k, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    #plt.ylim([0, 0.15])
    plt.legend()
    plt.show()

    myfile=open('gisette.dat', 'w')
    myfile.write('misclassTr missclassVal\n')
    for i in range(k.shape[0]):
        myfile.write(str(k[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()

def project3c():
    k=np.array([10,30,100,300])
    nk=len(k)
    misclassTr=np.zeros(nk)
    misclassVal=np.zeros(nk)
    for j in range(nk):
        xTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\MADELON\madelon_train.txt')
        yTr=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\MADELON\madelon_trainlabels.txt')
        xVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\MADELON\madelon_valid.txt')
        yVal=np.loadtxt('E:\study material\FSU\Fall2018\ML\AML_Fall2018\Data\MADELON\madelon_validlabels.txt')

        lr=FSA(xTr, yTr, xVal, yVal, k[j])
        misclassTr[j], misclassVal[j]=lr.train()

    figureIndex=1
    plt.figure(figureIndex)
    plt.plot(k, misclassTr, 'o-', label='train')
    plt.plot(k, misclassVal, 's-', label='test')
    plt.xlabel('num of features')
    plt.ylabel('miss classification ratio')

    #plt.ylim([0, 0.15])
    plt.legend()
    plt.show()

    myfile=open('madelon.dat', 'w')
    myfile.write('misclassTr missclassVal\n')
    for i in range(k.shape[0]):
        myfile.write(str(k[i])+' '
                +str(misclassTr[i])+' '+str(misclassVal[i])+'\n')
    myfile.close()


project3a()
#project3b()
#project3c()
    
    




    
