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

def G_K(X, Y, gamma):
  n = X.shape[0]
  m = Y.shape[0]
  xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
  yy = np.dot(np.sum(np.power(Y, 2), 1).reshape(m, 1), np.ones((1, n)))    
  r=np.exp(-gamma*(xx + yy.T - 2*np.dot(X, Y.T)))
  return r

def part_b():
  C = 1.0
  m,n = X.shape
  y= Y.reshape((m,1))
  # xx = np.dot(np.sum(np.power(xs, 2), 1).reshape(m, 1), np.ones((1, m)))  
  # G_K = np.exp(-0.001*(xx + xx.T - 2*np.dot(xs, xs.T)))
  start_time = time.time()
  P = matrix(np.multiply(G_K(X,X,0.001),np.dot(y,y.T)))
  q = matrix(-np.ones((m, 1)))
  G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
  h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
  A = matrix(y.transpose())
  b = matrix(0.0)
  sol = solvers.qp(P, q, G, h, A, b)
  alphas = np.array(sol['x'])
  S = (alphas > 1e-4).flatten()
  xs = X[S]
  ys = y[S]
  alphas = alphas[S]
  b = ys - np.sum(G_K(xs, xs, 0.001) * alphas * ys, axis=0)
  b = np.mean(b)
  end_time = time.time()
  t=end_time-start_time
  ty = G_K(xs,test_X, 0.001)
  pred_Y = np.sum(ty * alphas * ys, axis=0) + b
  cor=0
  Sl = part_a()
  for i in range(len(pred_Y)):
    if(pred_Y[i]>0.0 and test_Y[i]==1.0):
      cor=cor+1
    if(pred_Y[i]<0.0 and test_Y[i]==-1.0):
      cor=cor+1
  print(" part b ")
  print("accuracy = "+ str(cor/len(test_Y)))
  print("time taken for training = "+str(t))
  print("no of suppor vectors ="+str(len(alphas)))
  count=0
  for i in range(len(Sl)):
    if(Sl[i]==True and S[i]==True):
      count=count+1
  print("no of simialr vectors = "+str(count))
  c1=0
  i1=0
  while(c1<5):
    if(Sl[i1]):
      c1=c1+1
      plt.imshow(train['data'][i1].reshape((32,32,3)))
      plt.savefig("qa"+str(i1)+".png")
      plt.show(block=False)
      plt.pause(0.2)
    i1=i1+1


  c=0
  i=0
  while(c<5):
    if(S[i]):
      c=c+1
      plt.imshow(train['data'][i].reshape((32,32,3)))
      plt.savefig("qb"+str(i)+".png")
      plt.show(block=False)
      plt.pause(0.2)
    i=i+1

  return

part_b()