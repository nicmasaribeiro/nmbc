#!/usr/bin/env python3

import yfinance as yf
import numpy as np
import math

def f(M,t,mu):
	s = 0
	for i in range(t):
		value = (1/math.factorial(i))*M**i
		s += value
	return mu.T*s#np.sqrt(mu.T*s*mu)

t = yf.Tickers("AAPL IBM TSLA NVDA")
h = t.history()
print(h["Close"])
df = h['Close'].pct_change()[1:]

def count(df):
	p = 0
	n = 0
	for i in range(len(df)):
		if df[i] > 0:
			p+=1
		else:
			n+=1
	return p/(n+p)

c_aapl = count(df['AAPL'])
c_ibm = count(df['IBM'])
c_tsla = count(df['TSLA'])
c_nvda = count(df['NVDA'])

print('\n',c_aapl,c_ibm,c_tsla,c_nvda,'\n')
ls = [c_aapl,c_ibm,c_tsla,c_nvda]
D = np.eye(4)
for i in range(4):
	D[i,i] = ls[i]
print('\nD\n',np.linalg.inv(D))
print('\nD\n',D)

#print(df)
A = np.matrix(df)
print('\nA\n',A)

C = df.corr()
print('\nC\n',C)

Lattice = np.matrix([[1,1/2,3/4,1/2],
					 [1/2,1,1/2,1/2],
					 [3/4,1/2,1,3/4],
					 [1/2,1/2,3/4,1]])
#print(A*Lattice)
B = A*Lattice
G = B.T*B
print('\nG\n',G,'\n')
val,vec = np.linalg.eig(G)
print('\nval',np.round(val,3),'\n')
print('\nvec\n',vec,'\n')

mu = np.matrix(h["Close"])[-1]
print(mu)
print('\nexpectation\n',f(G,2,mu.T))

#px = np.matrix([[0,1,0,0],
#				[1,0,0,0],
#				[0,0,0,1],
#				[0,0,1,0]])
#print('\nbxObx\n',(B*px).T*(B*px),'\n')
#
#pz =  np.matrix([[1,0,-1,0],
#				 [0,-1,0,1],
#				 [-1,0,1,0],
#				 [0,1,0,-1]])
#
#pz =  np.matrix([[1,0,0,0],
#				 [0,-1,0,0],
#				 [0,0,1,0],
#				 [0,0,0,-1]])

#zGz = 100*pz*G*pz#10*
#print(zGz)
#AzGz = A*zGz
#print('\nAzGz\n',AzGz)
#AzGz = AzGz.T*AzGz
#print('\nAzGz\n',AzGz)
#b = np.matrix([ls]).T #1 for i in range(4)]
#print(b)
#X = np.linalg.inv(AzGz)*b
#print(X)
##print(h['Close'])
