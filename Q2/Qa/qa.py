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

def part_a():
  C = 1.0
  m,n = X.shape
  y= Y.reshape(m,1)
  X_p = y * X
  start_time = time.time()
  H = np.dot(X_p , X_p.T) * 1.
  P = matrix(H)
  q = matrix(-np.ones((m, 1)))
  G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
  h = matrix(np.vstack((np.zeros((m,1)), np.ones((m,1))*C)))
  #h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
  A = matrix(y.transpose())
  b = matrix(np.zeros(1))
  sol = solvers.qp(P, q, G, h, A, b)
  alphas = np.array(sol['x'])
  w = ((y * alphas).T @ X).reshape(-1,1)
  S = (alphas > 1e-4).flatten()
  b = y[S] - np.dot(X[S], w)
  b=np.mean(b)
  end_time = time.time()
  t=end_time-start_time
  pred_Y = np.dot(test_X,w)
  cor=0
  for i in range(len(pred_Y)):
    if(pred_Y[i,0]+b>0.0 and test_Y[i]==1.0):
      cor=cor+1
    if(pred_Y[i,0]+b<0.0 and test_Y[i]==-1.0):
      cor=cor+1
  print("Time taken = "+str(t))
  print("accuracy = "+str(cor/len(test_Y)))
  print("no of support vectors = "+str(len(alphas)))
  print("percentage of support vectors = "+str(len(alphas)/len(S)))
  plt.imshow(w.reshape(32,32,3))
  plt.savefig("weight.png")
  return S

part_a()