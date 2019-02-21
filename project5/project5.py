import numpy as np
import csv

def program5():
    data=np.genfromtxt('covtype.data', delimiter=',')
    print('data: ', data.shape)
    train=open('covtype_train.csv', 'w')
    trainW=csv.writer(train)
    test=open('covtype_test.csv', 'w')
    testW=csv.writer(test)

    m=data.shape[0]
    n=data.shape[1]
    mTrain=11340+3780
    mTest=565892
    #mTrain=10
    #mTest=5

    out=[]
    for i in range(n-1):
        vec='X'+str(i+1)
        out.append(vec)

    out.append('Y')

    trainW.writerow(out)
    testW.writerow(out)

    for i in range(mTrain):
        out=[]
        vec=data[i].astype(int)
        for j in range(vec.shape[0]-1):
           out.append(vec[j])
        out.append('C'+str(vec[-1]))
        trainW.writerow(out)

    for i in range(mTest):
        out=[]
        vec=data[i+mTrain].astype(int)
        for j in range(vec.shape[0]-1):
           out.append(vec[j])
        out.append('C'+str(vec[-1]))
        testW.writerow(out)
        
    train.close()
    test.close()

program5()
