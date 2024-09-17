#!/usr/bin/env python3
import requests 
import json

def get_dep(ticker):
	url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['depreciationAndAmortization']

def get_rev(ticker):
	url = "https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['ebitda']

def get_shares_two(ticker):
	url = "https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}/?period=quarter&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['numberOfShares']


def get_debt(ticker):
	url = "https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?limit=40&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['addTotalDebt']

def get_equity(ticker):
	url = "https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?limit=40&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['marketCapitalization']

def get_cash(ticker):
	url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['cashAndShortTermInvestments']

def get_ni(ticker):
	url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return data[0]['payoutRatioTTM']

