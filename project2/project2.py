import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def decision_tree(X, Y, XTest, YTest, kTrees, maxFeatures):
    rateTrain=np.zeros(kTrees.shape[0])
    rateTest=np.zeros(kTrees.shape[0])
    for i in range(kTrees.shape[0]):
        clf=RandomForestClassifier(n_estimators=kTrees[i], criterion='entropy',
                max_features=maxFeatures, random_state=13931)
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
    kTrees=np.array([3, 10, 30, 100, 300])
    train, test=decision_tree(X, Y, XTest, YTest, kTrees, 'sqrt')
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(kTrees, train, label='Error_train')
    plt.plot(kTrees, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('number of trees')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()

def problem_1b():
    X=np.loadtxt('madelon_train.data')
    Y=np.loadtxt('madelon_train.labels')
    XTest=np.loadtxt('madelon_valid.data')
    YTest=np.loadtxt('madelon_valid.labels')
    kTrees=np.array([3, 10, 30, 100, 300])
    train, test=decision_tree(X, Y, XTest, YTest, kTrees, 'log2')
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(kTrees, train, label='Error_train')
    plt.plot(kTrees, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('number of trees')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()

def problem_1c():
    X=np.loadtxt('madelon_train.data')
    Y=np.loadtxt('madelon_train.labels')
    XTest=np.loadtxt('madelon_valid.data')
    YTest=np.loadtxt('madelon_valid.labels')
    kTrees=np.array([3, 10, 30, 100, 300])
    train, test=decision_tree(X, Y, XTest, YTest, kTrees, None)
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(kTrees, train, label='Error_train')
    plt.plot(kTrees, test,  label='Error_test')
    #plt.figure(figureIndex)
    #figureIndex += 1
    #plt.plot(eps1[45:], err1[45:])
    plt.xlabel('number of trees')
    plt.ylabel('missclassification rate')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.show()


problem_1a()
#problem_1b()
#problem_1c()
