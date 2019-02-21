import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def decision_tree(X, Y, XTest, YTest):
    clf=LogisticRegression(C=0.001, max_iter=20000)
    clf=clf.fit(X, Y)
    y=clf.predict(X)
    yTest=clf.predict(XTest)
    rateTrain=1-np.mean(y==Y)
    rateTest=1-np.mean(yTest==YTest)
    print('train, test misclassification error: ', rateTrain,
            rateTest)
    return rateTrain, rateTest

def problem_1a():
    X=np.loadtxt('X.dat')
    Y=np.loadtxt('Y.dat')
    XTest=np.loadtxt('Xtest.dat')
    YTest=np.loadtxt('Ytest.dat')
    train, test=decision_tree(X, Y, XTest, YTest)
    
problem_1a()
#problem_1b()
#problem_1c()
