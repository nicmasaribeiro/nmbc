#!/usr/bin/env python3

#!/usr/bin/env python3
import yfinance as yf
import numpy as np
import pylcp as py

t = yf.Tickers("AAPL GM IBM APPS")
h = t.history()
df = h["Close"]
A = np.matrix(df.pct_change()[1:])

I = np.matrix([[1,1,-1,-1],
			   [1,-1,1,1],
			   [1,1,-1,-1],
			   [-1,1,-1,1]])	
N = A.T #np.matrix([ [.8,1],
#				[1.05,.95],
#				[1,1.2],
#				[.98,1]])
S = (I*N)
G = S*S.T
print(G)
e,v = np.linalg.eig(G)
print(e)


#M = np.kron(G, J)
#print('\nM')
#print(M)
#print('\n')



apz = np.matrix([[1,0,0,0],
				[0,-1,0,0],
				[0,0,1,0],
				[0,0,0,-1]])
apx = np.matrix([[0,1,0,0],
				[1,0,0,0],
				[0,0,0,1],
				[0,0,1,0]])
px = np.matrix([[0,1],[1,0]])
pz = np.matrix([[1,0],[0,-1]])

print('\nGx')
Gx = apx*G
print(Gx)
print('\nGz')
Gz = apz*G
print(Gz)
#print(np.linalg.eig(Gx)[1])
print('\n')
print(Gz)


print('\nHGx')
HGx = -10*np.matrix([[Gx[0,0]*Gx[1,1]-Gx[0,1]*Gx[1,0],Gx[0,2]*Gx[1,3]-Gx[0,3]*Gx[1,2]],
				[Gx[2,0]*Gx[3,1]-Gx[2,1]*Gx[3,0],Gx[2,2]*Gx[3,3]-Gx[2,3]*Gx[3,2]]])
print(HGx)
val,vec = np.linalg.eig(HGx)
print(val)
print(vec)
print(np.linalg.det(HGx))
#print(np.linalg.inv(HGx))
#K = np.kron(HGx, A)
#print(K)

print('\nHGxPz')
HGxPz = HGx*pz
print(HGxPz)
#print(np.linalg.eig(HGxPz))
#print(Gx[0,0]*Gx[1,1]-Gx[0,1]*Gx[1,0])
#print(Gx[2,0]*Gx[3,1]-Gx[2,1]*Gx[3,0])
#print(Gx[0,2]*Gx[1,3]-Gx[0,3]*Gx[1,2])
#print(Gx[2,2]*Gx[3,3]-Gx[2,3]*Gx[3,2])