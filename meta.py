#!/usr/bin/env python3
import requests
import json


t = input("ticker [+] << ").upper()

def get_beta(ticker):
	url = "https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['beta'])

def get_wacc(ticker):
	url = "https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['wacc'])/100

def get_taxes(ticker):
	url = "https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['taxRate'])/100

def get_costDebt(ticker):
	url = "https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['costofDebt'])/100

def get_interestCoverage(ticker):
	url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['interestCoverageTTM'])#/100

def get_price(ticker):
	url = "https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['price'])

print("beta >>\t",get_beta(t))
print("taxes >>\t",get_taxes(t))
print("costOfDebt >>\t",get_costDebt(t))
print("interestCoverage >>\t",get_interestCoverage(t))
print("wacc >>\t",get_wacc(t))
print("Price >>\t",get_price(t))