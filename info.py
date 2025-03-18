

import requests
import json

def asset_info(t):
	def get_info(ticker):
		url = "https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		response = requests.request("GET", url)
		data = json.loads(response.text)
		return data
	mk = get_info(t)['marketCap']
	beta = get_info(t)['beta']
	rng = get_info(t)['range']
	change = get_info(t)['change']
	change_percent = get_info(t)['changePercentage']
	volume = get_info(t)['volume']
	avg_volume = get_info(t)['averageVolume']
	ceo = get_info(t)['ceo']
	industry = get_info(t)['industry']
	website = get_info(t)['website']
	img = get_info(t)['image']
	description = get_info(t)['description']

	return (mk,beta,rng,change,change_percent,volume,avg_volume,ceo,industry,website,img,description)
# print(asset_info('AAPL'))