

import requests
import json

def asset_info(t):
	def get_info(ticker):
		url = "https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey=67824182044bfc7088c8b3ee21824590".format(ticker=ticker)
		response = requests.request("GET", url)
		data = json.loads(response.text)
		return data
	mk = get_info(t)[0]['marketCap']
	beta = get_info(t)[0]['beta']
	rng = get_info(t)[0]['range']
	change = get_info(t)[0]['change']
	change_percent = get_info(t)[0]['changePercentage']
	volume = get_info(t)[0]['volume']
	avg_volume = get_info(t)[0]['averageVolume']
	ceo = get_info(t)[0]['ceo']
	industry = get_info(t)[0]['industry']
	website = get_info(t)[0]['website']
	img = get_info(t)[0]['image']
	description = get_info(t)[0]['description']

	return (mk,beta,rng,change,change_percent,volume,avg_volume,ceo,industry,website,img,description)
# print(asset_info('AAPL'))