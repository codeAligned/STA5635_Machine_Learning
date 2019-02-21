import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import tree

def decision_tree(X, Y, XTest, YTest, maxDepth):
    rateTrain=np.zeros(maxDepth.shape[0])
    rateTest=np.zeros(maxDepth.shape[0])
    for i in range(maxDepth.shape[0]):
        clf=tree.DecisionTreeClassifier(criterion='entropy',
                max_depth=maxDepth[i] )
        clf=clf.fit(X, Y)
        y=clf.predict(X)
        yTest=clf.predict(XTest)
        rateTrain[i]=1-np.mean(y==Y)
        rateTest[i]=1-np.mean(yTest==YTest)
        print('train, test misclassification error: ', rateTrain[i],
                rateTest[i])
    return rateTrain, rateTest

def problem_1a():
    X=np.loadtxt('madelon_train.data')
    Y=np.loadtxt('madelon_train.labels')
    XTest=np.loadtxt('madelon_valid.data')
    YTest=np.loadtxt('madelon_valid.labels')
    maxDepth=np.arange(12)+1
    train, test=decision_tree(X, Y, XTest, YTest, maxDepth)
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(maxDepth, train, label='Error_train')
    plt.plot(maxDepth, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('max depth')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

def problem_1b():
    X=genfromtxt('wilt_train.csv', delimiter=',')
    Y=np.loadtxt('wilt_train.labels')
    XTest=genfromtxt('wilt_test.csv', delimiter=',')
    YTest=np.loadtxt('wilt_test.labels')
    maxDepth=np.arange(10)+1
    train, test=decision_tree(X, Y, XTest, YTest, maxDepth)
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(maxDepth, train, label='Error_train')
    plt.plot(maxDepth, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('max depth')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

def problem_1c():
    X=np.loadtxt('gisette_train.data')
    Y=np.loadtxt('gisette_train.labels')
    XTest=np.loadtxt('gisette_valid.data')
    YTest=np.loadtxt('gisette_valid.labels')
    maxDepth=np.arange(6)+1
    train, test=decision_tree(X, Y, XTest, YTest, maxDepth)
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(maxDepth, train, label='Error_train')
    plt.plot(maxDepth, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('max depth')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.4])
    plt.legend()
    plt.show()

problem_1a()
problem_1b()
problem_1c()
