#!/usr/bin/env python3
from scipy.misc import derivative
import scipy as sp
import numpy as np
from scipy.integrate import quad, dblquad


dt = 1/52

ri,di,si = .03,.04,10
rk,dk,sk = .033,.054,10

pi = lambda x:sp.stats.poisson(dt).cdf(x)
Bi = lambda x: si * (x - np.exp(-ri*x))
fi = lambda t,T: np.exp((T-t)*(ri-di)) * Bi(T)
ii = lambda x,y: fi(x,y)*pi(x)
ui = lambda t,T: dblquad(ii,t,T,lambda x: t, lambda x: T)[0]
print(ui(0,3))

pk = lambda x: sp.stats.norm(0,dt).pdf(x)
Bk = lambda x: sk * (x - np.exp(-rk*x))
fk = lambda t, T: np.exp((T-t)*(rk-dk)) * Bk(T)
ik = lambda x,y: fk(x,y)*pk(x)
uk = lambda t,T: dblquad(ik,t,T,lambda x: t, lambda x: T)[0]
print(uk(0,3))

p0 = lambda x,y: (1/np.cosh(x))*np.tanh(y)
i_p0 =  lambda t,T : dblquad(p0, t, T, lambda x: t, lambda x: T)[0]
Bik = lambda t,T: uk(t,T) + ui(t,T) + i_p0(t,T)
print(Bik(0,2))

i1 = lambda x,y: p0(x,y)*Bi(x)*Bk(y)
i2 = quad(Bk, 0, 1)[0]
i3 = quad(Bi, 0, 1)[0]
Cik = lambda t,T : dblquad(i1, t	, T,lambda x: t, lambda x: T)[0]**2 - i2 * i3
print(Cik(0,1))

#
#def dCikdt(t,T):
#	V_func = lambda t,T : Cik(t,T)
#	return derivative(V_func, t, dx=1e-5)
#print(dCikdt(1, 2))