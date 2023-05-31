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

X=[]
Y=[]
for i in range(len(train['data'])):
    if(train['labels'][i][0]==2):
        X.append(train['data'][i].flatten())
        Y.append(1.0)
    if(train['labels'][i][0]==1):
        X.append(train['data'][i].flatten())
        Y.append(-1.0)
X=np.array(X)
X=X/255
Y = np.array(Y)
Y=Y.reshape((X.shape[0],1))


test = pickle.load(open(test_path+'/test_data.pickle','rb'))
test_X=[]
test_Y=[]
for i in range(len(test['data'])):
    if(test['labels'][i][0]==2):
        test_X.append(test['data'][i].flatten())
        test_Y.append(1.0)
    if(test['labels'][i][0]==1):
        test_X.append(test['data'][i].flatten())
        test_Y.append(-1.0)
test_X = np.array(test_X)
test_Y = np.array(test_Y)
test_X = test_X/255
print(test_X)
print(test_Y)


def part_c():
  m = len(Y)
  start_time1 = time.time()
  cs1 = SVC(kernel = 'linear',C=1)
  cs1.fit(X,Y)
  end_time1 =time.time()
  pred1 = cs1.predict(test_X)
  cor=0
  for i in range(len(test_Y)):
    if(test_Y[i]==pred1[i]):
      cor=cor+1
  start_time2 = time.time()
  cs2 = SVC(kernel = 'rbf',gamma = 0.001,C=1)
  cs2.fit(X,Y)
  end_time2 =time.time()
  pred2 = cs2.predict(test_X)
  cor1 = 0
  for i in range(len(test_Y)):
    if(test_Y[i]==pred2[i]):
      cor1 = cor1+1
  print("Time taken for linear = "+str(end_time1-start_time1))
  print("Accuracy of linear= "+str(cor/len(test_Y)))
  s1=cs1.support_
  dic ={}
  for i in range(len(s1)):
    dic[s1[i]]=1
  s2 = cs2.support_
  c=0
  for i in range(len(s2)):
    if(dic.get(s2[i])!=None):
      c=c+1
  print("no of supporting vectors in linear  ="+str(cs1.n_support_[0]+cs1.n_support_[1]))
  print("no of supporting vectors in gaussian = "+str(cs2.n_support_[0]+cs2.n_support_[1]))
  print("time taken for gaussian =  "+str(end_time2-start_time2))
  print("accuracy of gaussian = "+str(cor1/len(test_Y)))
  print("No of common support vectors = "+str(c))

part_c()