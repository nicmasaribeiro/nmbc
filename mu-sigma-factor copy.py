#!/usr/bin/env python3

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import datetime as dt
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
ticker = input('[=] ticker [=]\t').upper()
t = yf.Tickers(f"{ticker} ^GSPC ^NDX ^DJI ^VIX")
h = t.history(start='2015-1-1',end=dt.date.today(),interval='1d')#["Close"]
df = h["Close"] 
ret = df.pct_change()[1:]
mu = ret.rolling(3).mean().dropna()
sig = ret.rolling(3).std().dropna()
X = np.hstack((mu,sig))
y = np.matrix(ret[ticker][2:].dropna()).T
lr  = LinearRegression(fit_intercept=True)
fit = lr.fit(X, np.asarray(y))
x = fit.coef_
print("Score\t",fit.score(X,np.asarray(y)))
factor_matrix = np.matrix([X.T[0,-1],X.T[1,-1],X.T[2,-1],X.T[3,-1],X.T[4,-1],X.T[5,-1],X.T[6,-1],X.T[7,-1],X.T[8,-1],X.T[9,-1]]).T
pred = x@factor_matrix
print("Predicted Change...\t",pred.item()*100,'%')
