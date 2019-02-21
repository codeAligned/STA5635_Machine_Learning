import numpy as np
import matplotlib.pyplot as plt

class logisticRegression:
    def __init__(self, xTr, yTr, xVal, yVal,
            nIter=500, eta=0.0025, k=10, s=0.001, miu=100.,
            classes=7, lam=0.001):
        self.xTr=xTr
        # caution, still a view, change self.yTr will change yTr and
        # propogate outside this function!!!!
        self.yTr=yTr
        self.xVal=xVal
        self.yVal=yVal
        self.s=s
        self.miu=miu
        self.nIter=nIter
        self.k=k
        self.eta=eta
        self.classes=classes
        self.lam=lam

        self.preProcess()
        # M is based on the duplicate features removed version
        self.M=self.xTr.shape[1]
        self.w=np.zeros((self.xTr.shape[1], self.classes-1))

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

        self.yTr-=1.
        self.yVal-=1.
        print('samples', self.xTr.shape)
        print('finish preprocessing data')

    def gradient(self):
        u=self.xTr.dot(self.w)
        zeros=np.zeros((self.xTr.shape[0], 1))
        u=np.append(u, zeros, axis=1)
        #print(u[:5])
        #print(self.yTr[:5])
        uy=np.choose(self.yTr.astype(int), u.T)
        #print(uy[:5])

        deltaU=(uy-u.T).T-1.
        temp=2.*deltaU/(1.+deltaU*deltaU)
        temp[deltaU>0.]=0.
        c=-np.copy(temp)
        temp1=(u.T==uy).T
        temp2=np.sum(temp, axis=1)+1.
        #print('before modification, c', c[:5])
        c[range(c.shape[0]), self.yTr.astype(int)]=np.sum(temp, axis=1)+1.
        #print('after modification, c', c[:5])
        c=c[:, :-1] #cut last column
        c=c.T
        grad=(c.dot(self.xTr)).T+2.*self.lam*self.w
        return grad

    def update(self, i):
        grad=self.gradient()
        self.w-=self.eta*grad
        self.fsa(i)

    def fsa(self, i):
        mi=self.k+(self.M-self.k)*max(0.,
                (self.nIter-2*i)/(2*i*self.miu+self.nIter))
        mi=int(mi)
        print(i, 'mi', mi)
        wAbs=np.absolute(self.w)
        wAbsSort=np.argsort(np.sum(wAbs, axis=1))
        wAbsSort=wAbsSort[-mi:]
        self.w=self.w[wAbsSort, :]
        self.xTr=(self.xTr.T[wAbsSort]).T
        self.xVal=(self.xVal.T[wAbsSort]).T

    def train(self):
        loss=np.zeros(self.nIter)
        misclassTr=np.zeros(self.nIter)
        misclassVal=np.zeros(self.nIter)
        for i in range(self.nIter):
            self.update(i)
            u=self.xTr.dot(self.w)
            zeros=np.zeros((self.xTr.shape[0], 1))
            u=np.append(u, zeros, axis=1)
            uy=np.choose(self.yTr.astype(int), u.T)
            deltaU=(uy-u.T).T-1.
            loss0=np.log(1.+deltaU*deltaU)
            loss0[deltaU>0]=0.
            loss[i]=np.sum(loss0)-self.xTr.shape[0]*np.log(2.)

            pred=np.argmax(u, axis=1)
            misclassTr[i]=1.-np.mean(pred==self.yTr)

            u=self.xVal.dot(self.w)
            zeros=np.zeros((self.xVal.shape[0], 1))
            u=np.append(u, zeros, axis=1)
            pred=np.argmax(u, axis=1)
            misclassVal[i]=1.-np.mean(pred==self.yVal)
            print(i, 'loss, miss classification err in train, val: ',
                    loss[i], misclassTr[i], misclassVal[i])
        return loss, misclassTr, misclassVal

def project7a():
    xTr=np.loadtxt('sat.trn')
    yTr=xTr[:, -1]
    xTr=xTr[:, :-1]

    xVal=np.loadtxt('sat.tst')
    yVal=xVal[:, -1]
    xVal=xVal[:, :-1]
    print('finish loading data')

    ks=[5, 9, 18, 27, 36]
    etas=[0.00004, 0.00004, 0.00006, 0.0001, 0.00005]
    misclassErrTr=np.zeros(len(ks))
    misclassErrVal=np.zeros(len(ks))
    figureIndex=0
    for i in range(len(ks)):
        lr=logisticRegression(np.copy(xTr), np.copy(yTr), 
                              np.copy(xVal),np.copy(yVal), 
                              eta=etas[i], k=ks[i])
        loss, misclassTr, misclassVal=lr.train()
        #index=np.argsort(misclassVal)
        #misclassErrTr[i]=misclassTr[index[0]]
        #misclassErrVal[i]=misclassVal[index[0]]
        misclassErrTr[i]=misclassTr[-1]
        misclassErrVal[i]=misclassVal[-1]

        plt.figure(figureIndex)
        figureIndex += 1
        plt.plot(np.arange(loss.shape[0]), misclassTr, label='train')
        plt.plot(np.arange(loss.shape[0]), misclassVal, label='test')
        plt.xlabel('iteration')
        plt.ylabel('miss classification ratio')
        plt.legend()

        plt.figure(figureIndex)
        figureIndex += 1
        plt.plot(np.arange(loss.shape[0]), loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')

    myfile=open('satimage.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(misclassErrTr[i])+
                                ' '+str(misclassErrVal[i])+'\n')
    myfile.close()
    plt.show()

def project7b():
    x=np.genfromtxt('covtype.data', delimiter=',')
    xTr=x[:(11340+3780), :-1]
    yTr=x[:(11340+3780), -1]
    xVal=x[(11340+3780):, :-1]
    yVal=x[(11340+3780):, -1]
    test=False
    if test:
        xTr=x[:10, :-1]
        yTr=x[:10, -1]
        xVal=x[10:20, :-1]
        yVal=x[10:20, -1]
    print('finish loading data')
    ks=[5, 9, 18, 27, 36]
    etas=[0.00005, 0.00004, 0.00004, 0.00004, 0.00004]
    misclassErrTr=np.zeros(len(ks))
    misclassErrVal=np.zeros(len(ks))
    figureIndex=0
    for i in range(len(ks)):
        lr=logisticRegression(np.copy(xTr), np.copy(yTr), 
                              np.copy(xVal),np.copy(yVal), 
                              eta=etas[i], k=ks[i])
        loss, misclassTr, misclassVal=lr.train()
        #index=np.argsort(misclassVal)
        #misclassErrTr[i]=misclassTr[index[0]]
        #misclassErrVal[i]=misclassVal[index[0]]
        misclassErrTr[i]=misclassTr[-1]
        misclassErrVal[i]=misclassVal[-1]

        plt.figure(figureIndex)
        figureIndex += 1
        plt.plot(np.arange(loss.shape[0]), misclassTr, label='train')
        plt.plot(np.arange(loss.shape[0]), misclassVal, label='test')
        plt.xlabel('iteration')
        plt.ylabel('miss classification ratio')
        plt.legend()

        plt.figure(figureIndex)
        figureIndex += 1
        plt.plot(np.arange(loss.shape[0]), loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')

    myfile=open('covtype.dat', 'w')
    for i in range(len(ks)):
        myfile.write(str(ks[i])+' '+str(misclassErrTr[i])+
                                ' '+str(misclassErrVal[i])+'\n')
    myfile.close()
    plt.show()

def draw():
    covtype=np.loadtxt('covtype.dat')
    satimage=np.loadtxt('satimage.dat')
    figureIndex = 0

    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(satimage[:,0], satimage[:,1], 'o-', label='train')
    plt.plot(satimage[:,0], satimage[:,2], 's-', label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')
    plt.legend()

    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(covtype[:,0], covtype[:,1], 'o-', label='train')
    plt.plot(covtype[:,0], covtype[:,2], 's-', label='test')
    plt.xlabel('k')
    plt.ylabel('misclassification ratio')
    plt.legend()

    plt.show()

#project7a()
#project7b()
draw()
    
    




    
