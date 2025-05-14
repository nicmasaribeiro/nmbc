#67824182044bfc7088c8b3ee21824590
import json
import requests as rq
import pandas as pd

def get_price(tick):	
	url = f'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={tick}&apikey=67824182044bfc7088c8b3ee21824590'
	result = rq.get(url).text
	data = json.loads(result)
	df = pd.DataFrame(data)
	return df



# RES = get_price('ADAUSD')
# print(RES)