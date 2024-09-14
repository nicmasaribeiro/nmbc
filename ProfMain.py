#!/usr/bin/env python3

import requests
import json
import pandas as pd
class ProfiMain:
	
	def get_workingCapital(self,ticker):
		url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		
		response = requests.request("GET", url)
		data = json.loads(response.text)
		# print(data[0]['totalCurrentLiabilities'])
		df =[]
		for i in range(0,5):
			df.append(data[i]['totalAssets'])
		# print(data[0]['sellingGeneralAndAdministrativeExpenses'])
		df2 = []
		for i in range(0, 5):
			df2.append(data[i]['totalCurrentLiabilities'])
		# print(data[0]['sellingGeneralAndAdministrativeExpenses'])
		df3 = [df[i]-df2[i] for i in range(0,5)]
		
#		print('Capital Employed',df3)
		return df3
	#
	# print(get_workingCapital('AAPL'))
	
	def get_operatingIncome(self,ticker):
		url = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		
		response = requests.request("GET", url)
		data = json.loads(response.text)
		# print(data[0]['totalCurrentLiabilities'])
		df =[]
		for i in range(0,5):
			df.append(data[i]['operatingIncome'])
#		print("operatingIncome",df)
		return df
	# print(get_operatingIncome('AAPL'))
	
	def roc(self,ticker):
		roc = []
		for i in range(0,4):
			r = self.get_operatingIncome(ticker)[i+1]/self.get_workingCapital(ticker)[i]
			roc.append(r)
		return (roc)
	
#s = ProfiMain()
#print(s.roc('APA'))
