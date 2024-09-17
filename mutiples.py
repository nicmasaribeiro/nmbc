#!/usr/bin/env python3
import requests
import json
from ProfMain import ProfiMain
import numpy as np
from api import get_ni
from Roe import ProfiMainEquity
import numpy as np
from api import *

	
def price_to_book(roe,payout,wacc,g):
	return (roe*payout)/(wacc-g)

def ev_bookCapital(roc,g,wacc):
	return (roc-g)/(wacc-g)

def implied_div_growth_rate(ke,dps,market_price):
	return ke - (dps/market_price)

def fcff(ebi,t,g,roc):
	return ebi-(g/roc)*ebi
