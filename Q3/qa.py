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

def part_3_a():
  t=0
  V=[]
  cf = []
  for i in range(5):
    cf.append([])
    for j in range(5):
      cf[i].append(0)
  for i in range(len(test['data'])):
    V.append([])
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
        Y_p=Y_p.reshape((X_p.shape[0],1))
        C_p = 1.0
        m_p,n_p = X_p.shape
        y_p= Y_p.reshape((m_p,1))
        start_time = time.time()
        P_p = matrix(np.multiply(G_K(X_p,X_p,0.001),np.dot(y_p,y_p.T)))
        q_p = matrix(-np.ones((m_p, 1)))
        G_p = matrix(np.vstack((np.eye(m_p)*-1,np.eye(m_p))))
        h_p = matrix(np.hstack((np.zeros(m_p), np.ones(m_p) * C_p)))
        A_p = matrix(y_p.transpose())
        b_p = matrix(0.0)
        sol_p = solvers.qp(P_p, q_p, G_p, h_p, A_p, b_p)
        alphas_p = np.array(sol_p['x'])
        S = (alphas_p > 1e-4).flatten()
        xs = X_p[S]
        ys = Y_p[S]
        ys=np.array(ys)
        ys=ys.reshape((xs.shape[0],1))
        alphas_p = alphas_p[S]
        hh = G_K(xs, xs, 0.001)
        p_b = ys - np.sum( hh* alphas_p * ys, axis=0)
        p_b = np.mean(p_b)
        end_time = time.time()
        t=t+end_time-start_time
        test_x = test['data'].reshape((len(test['data'])),3072)/255
        tt = G_K(xs,test_x, 0.001) * alphas_p
        tt=tt*ys
        pred = np.sum(tt, axis=0) + p_b
        for k in range(len(test['data'])):
          if(pred[k]>0):
            V[k].append(j)
          else:
            V[k].append(i)
  cor=0
  count=0
  for i in range(len(test['data'])):
    cf[test['labels'][i][0]][md(V[i])]=cf[test['labels'][i][0]][md(V[i])]+1
    if(md(V[i])==test['labels'][i][0]):
      cor=cor+1
    count=count+1
  print("accuracy = "+str(cor/count))
  print("time taken for training = "+str(t))
  print(cf)
  return

part_3_a()