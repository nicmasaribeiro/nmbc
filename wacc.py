#!/usr/bin/env python3

class Rates:
	def __init__(self,D,E,t):
		self.D = D
		self.E = E
		self.t = t
		
	def wacc(self,rd,re):
		return (self.D/(self.E+self.D))*(1-self.t)*(rd) + (self.E/(self.E+self.D))*re
	
	def debt_beta(self,credit_spread,erp):
		return credit_spread/erp
	
	def unleverd_beta(self,regression_beta):
		return regression_beta/(1+(1-self.t)*(self.D/self.E))
	
	def re(self,reg_beta,rf,erp,credit_spread):
		bl = self.unleverd_beta(reg_beta)*(1+(1-self.t)*(self.D/self.E)) - self.debt_beta(credit_spread, erp) *(1-self.t)*(self.D/self.E)
		return rf+bl*erp
		
#		
#s = Rates(2976, 4963, 0.25)
#ke = s.re(1.98, 0.035, 0.047, 0.0179)
#print(ke)
#print(s.wacc(0.0585, ke))
##print(s.re(0.5, 0.04, 0.05))
#s = Rates(100, 50, 0.35)
#ke = s.re(1, 0.03, 0.06)
#wacc =s.wacc(0.035, ke)
#print(wacc)
##ke = re(100, 200,0.35, 1, 0.04, 0.06)
##print(wacc(100, 200, 0.35, 0.05, ke))
