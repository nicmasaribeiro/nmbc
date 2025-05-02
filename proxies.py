import requests
import yfinance as yf
import random

class YFinanceProxyWrapper:
	def __init__(self, proxy_list, rotate=True, max_retries=3, delay=2):
		self.proxy_list = proxy_list
		self.rotate = rotate
		self.max_retries = max_retries
		self.delay = delay
		self._original_get = requests.get
		self._patch_requests()
		
	def _get_proxy(self):
		return random.choice(self.proxy_list)
	
	def _patched_get(self, *args, **kwargs):
		for attempt in range(self.max_retries):
			proxy = self._get_proxy() if self.rotate else self.proxy_list[0]
			try:
				kwargs["proxies"] = {"http": proxy, "https": proxy}
				return self._original_get(*args, **kwargs)
			except Exception as e:
				print(f"Proxy failed: {proxy} ({e}), retrying...")
				time.sleep(self.delay)
		raise Exception("All proxy attempts failed")
		
	def _patch_requests(self):
		requests.get = self._patched_get
		
	def restore(self):
		requests.get = self._original_get
		
	def fetch(self, ticker, **kwargs):
		return yf.Ticker(ticker).history(**kwargs)
	
	