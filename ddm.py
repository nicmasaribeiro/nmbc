

from api import *
from mutiples import implied_div_growth_rate
from wacc import Rates


def get_price(ticker):
	url = "https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['price'])


def get_beta(ticker):
	url = "https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
	response = requests.request("GET", url)
	data = json.loads(response.text)
	return float(data[0]['beta'])


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

#def generate_csv():
	

#t = input("ticker [+] << ").upper()
#taxes = get_taxes(t)#float(input("tax [+] << "))
#reg = get_beta(t) #float(input("reg beta [+] << "))
#rf = 0.0415#float(input("rf [+] << "))
#erp = 0.0472 #float(input("equity risk premium[+] << "))
#print("Interest Coverage {=}\t",get_interestCoverage(t))
#cs = float(input("credit spread [+] << "))
#rd = get_costDebt(t)#float(input("cost of debt [+] << "))
#debt   = get_debt(t)
#equity = get_equity(t)
#
#
#rate = Rates(debt, equity, taxes)
#ke = rate.re(reg, rf, erp, cs)
#print("[+] Cost of Equity [+] >> ",ke)
#costOfCapital = rate.wacc(rd,ke)
#print("[+] WACC [+] >> ",costOfCapital)
#
#
#current_price = get_price(t) #float(input("[+] Market Price [+] >> "))
#print("[+] Price [+] >>\t",current_price)
#div_growth=implied_div_growth_rate(ke, get_dps(t), current_price)
#print("[+] Div Growth [+] >> ",div_growth)
#ddm = (get_dps(t)*(1+div_growth))/(ke - div_growth)
#print("[+] DDM [+] >> ",ddm)
