#!/usr/bin/env python3

#!/usr/bin/env python3
import requests
import numpy as np
from dcf4 import DCF 
from wacc import Rates #>)
from ProfMain import ProfiMain
import json
import pandas as pd

#ticker= input(">> ticker\t")

def get_ni(ticker):
	url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['payoutRatioTTM']

def get_rev(ticker):
	url = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['netIncome']

def get_shares(ticker):
	url = "https://financialmodelingprep.com/api/v4/shares_float?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['outstandingShares']

def get_debt(ticker):
	url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['netDebt']

def get_equity(ticker):
	url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['totalEquity']

def get_cash(ticker):
	url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['cashAndShortTermInvestments']

def get_interest_expense(ticker):
	url = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	df =[]
	for i in range(0,5):
		df.append(data[i]['interestExpense'])
		
	return np.mean(df)

def get_capex(ticker):
	url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	df =[]
	for i in range(0,5):
		df.append(data[i]['capitalExpenditure'])
		
	return np.mean(df)

def get_dep(ticker):
	url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data[0]['depreciationAndAmortization']

def get_dps(ticker):
	url = "https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	
	response = requests.request("GET", url)
	data = json.loads(response.text)
	
	return data["historical"][0]['adjDividend']

