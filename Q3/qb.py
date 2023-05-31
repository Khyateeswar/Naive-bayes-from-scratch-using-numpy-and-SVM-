import numpy as np
import sys
import pickle
from cvxopt import matrix 
from cvxopt import solvers 
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt


train_path = str(sys.argv[1])
test_path = str(sys.argv[1])
# train_path ='train_data.pickle'
# test_path = 'test_data.pickle'
train = pickle.load(open(train_path+'/train_data.pickle','rb'))
test = pickle.load(open(test_path+'/test_data.pickle','rb'))

def G_K(X, Y, gamma):
  n = X.shape[0]
  m = Y.shape[0]
  xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
  yy = np.dot(np.sum(np.power(Y, 2), 1).reshape(m, 1), np.ones((1, n)))    
  r=np.exp(-gamma*(xx + yy.T - 2*np.dot(X, Y.T)))
  return r

def md(l):
  dic = {}
  for i in range(5):
    dic[i]=0;
  for i in range(len(l)):
    dic[l[i]]=dic[l[i]]+1
  m =0
  for i in range(1,5):
    if(dic[i]>dic[m]):
      m=i
  return m

def part_3_b():
  V=[]
  t=0
  for i in range(len(test['data'])):
    V.append([])
  cf = []
  for i in range(5):
    cf.append([])
    for j in range(5):
      cf[i].append(0)
  for i in range(5):
    for j in range(5):
      if(j>i):
        X_p=[]
        Y_p=[]
        for k in range(len(train['data'])):
          if(train['labels'][k][0]==j):
            X_p.append(train['data'][k].flatten())
            Y_p.append(1.0)
          if(train['labels'][k][0]==i):
            X_p.append(train['data'][k].flatten())
            Y_p.append(-1.0)
        X_p=np.array(X_p)
        X_p=X_p/255
        Y_p = np.array(Y_p)
        Y_p.reshape((X_p.shape[0],1))
        start_time = time.time()
        cs2 = SVC(kernel = 'rbf',gamma = 0.001,C=1)
        cs2.fit(X_p,Y_p)
        end_time = time.time()
        t=t+end_time-start_time
        for k in range(len(test['data'])):
          if(cs2.predict((test['data'][k]).reshape((1,3072))/255)==1):
            V[k].append(j)
          elif(cs2.predict((test['data'][k]).reshape((1,3072))/255)==-1):
            V[k].append(i)
          else:
            print("Something is wrong")
        print(str(i)+" "+str(j)+" done")
  cor=0
  count=0
  ic=0
  for i in range(len(test['data'])):
    cf[test['labels'][i][0]][md(V[i])]=cf[test['labels'][i][0]][md(V[i])]+1
    if(md(V[i])==test['labels'][i][0]):
      #print("Correct Prediction")
      cor=cor+1
    count=count+1
    if(ic<10):
      if(md(V[i])==4 and test['labels'][i][0]==2):
        plt.imshow(test['data'][i].reshape((32,32,3)))
        plt.savefig("misc"+str(ic)+".png")
        plt.show(block=False)
        plt.pause(0.2)
        ic=ic+1
  print("accuracy = "+str(cor/count))
  print("Time taken fro training = "+str(t))
  print(cf)
  return

part_3_b()