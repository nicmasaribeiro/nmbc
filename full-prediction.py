#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.linear_model import LinearRegression

ticker = 'APPS' #input('enter ticker << \t').upper()
data = yf.Tickers("{t} ^IRX ^TNX ^TYX ^FVX ^VIX ^GSPC ^NDX ^DJI".format(t=ticker))
history = data.history(start ='2015-1-1',end =dt.date.today() ,interval='3mo')
df = history['Close'].dropna()

target = df[['^IRX','^FVX','^TNX','^TYX','^VIX','^GSPC','^NDX','^DJI']].pct_change()[1:]
target['lag'] = df[ticker].pct_change()[1:]

A = np.matrix(target)
b = np.matrix([df[ticker].pct_change()[1:]]).T
reg = LinearRegression().fit(np.array(A[:-2]), np.array(b[2:]))
x = reg.coef_
score = reg.score(np.array(A[:-2]), np.array(b[2:]))
#print('\nScore\n',np.round(reg.score(np.array(A[:-2]), np.array(b[2:])),3))

# These are the trailing 1m return im the predictor variables
irx = target['^IRX'][-1]
fvx = target['^FVX'][-1]#0.0256#12.95/100
tnx = target['^TNX'][-1]#0.0272#9.28/100
tyx = target['^TYX'][-1]#0.0271# 4/100
vix = target['^VIX'][-1]#-.3#10.2/100
snp = target['^GSPC'][-1]#-0.00672#3.91/100
ndx = target['^NDX'][-1]#-0.027#4.33/100
dji = target['^DJI'][-1]#0.1#2.47/100
lag = df[ticker].pct_change()[1:][-2]
target = target.dropna()

f = irx*x[0][0] + fvx*x[0][1] + tnx*x[0][2] + tyx*x[0][3] + vix*x[0][4] + snp*x[0][5] + ndx*x[0][6] + dji*x[0][7] + lag*x[0][8]
def fuc(s0,r,t):
	return s0*np.exp(r*t)
expected_change = f
expected_price = (1+f)*df[ticker][-1]
print(score,expected_change,expected_price)
