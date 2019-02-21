import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('errors_time.dat')
time=data[:,1]
error=data[:,0]*0.01
algs=['decision tree', 'Random Forest_100', 'Random Forest_300', 
      'Logistic Regression', 'Naive Bayes', 'Adaboost_J48',
      'Adaboost_100 trees', 'LogitBoost_10 stumps', 
      'LogitBoost_100 stumps', 'LogitBoost_100 stumps_trimming',
      'LogitBoost_25 M5P', 'SVM']
fig, ax=plt.subplots()
#ax.scatter(time, error, 'ro')
ax.scatter(time, error)
for i, txt in enumerate(algs):
    ax.annotate(txt, (time[i], error[i]), fontsize=9)
#ax.set_xscale('log')
plt.xlabel('time')
plt.ylabel('error')
plt.show()

