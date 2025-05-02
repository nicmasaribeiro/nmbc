from flask import Flask, render_template,send_from_directory, request, redirect, abort, jsonify,sessions, Response, url_for,send_file,render_template_string,flash
from flask import Blueprint
from flask_caching import Cache
import asyncio
import socket
import os
import html
import os
import statsmodels.api as sm
from quart import Quart
from web3 import Web3
import os
import csv
import random
import subprocess as sp
import xml.etree.ElementTree as ET
import scipy
from flask_executor import Executor
import xml.dom.minidom
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.embed import file_html, components
from bokeh.resources import CDN
from geom_forecast import GeometricBrownianMotion
import matplotlib.pyplot as plt
import datetime as dt
import base64
from bs import black_scholes
import requests
from models import *
from sqlalchemy import create_engine
from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
from cryptography.hazmat.primitives import serialization
from sqlalchemy.orm import sessionmaker
from werkzeug.utils import secure_filename
from sqlalchemy import delete
import json
import yfinance as yf
import stripe
from flask_login import current_user, login_required, login_user
import time
from hashlib import sha256
from bc import * 
from subprocess import Popen, PIPE
import openai
import threading
from sklearn.linear_model import LinearRegression 
from get_fundamentals import *
from ddm import *
from algo import stoch_price
from pricing_algo import derivative_price
from p2p import P2PNode
import fastapi
from bc2 import NodeBlockchain
from Transaction import Transaction
from p2pnode import P2PNode
import socket
import websocket as ws
import asyncio
import threading
import plotly.express as px
import plotly
from stoch_greeks import calculate_greeks
from scipy.integrate import quad
from scipy.stats import poisson
from sqlalchemy.orm import scoped_session, sessionmaker
import uuid
import logging
import schedule # apscheduler.schedulers.background import BackgroundScheduler
import threading
from twilio.rest import Client
from flask import session
from arch import arch_model
from celery import Celery
import ssl
from flask_login import LoginManager
import redis
from sklearn.decomposition import PCA
import subprocess
from kaggle_ui import kaggle_bp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from kaggle_ui import register_template_filters
from sequential_bp import sequential_bp
from proxies import YFinanceProxyWrapper


proxy_list = [
"http://24.249.199.12:4145",
"http://45.77.67.203:8080",
"http://138.68.60.8:3128"
"http://50.174.7.157:80",
"http://172.66.43.12:80",
"http://133.18.234.13:80",
"http://81.169.213.169:8888",
"http://194.158.203.14:80"
]
	

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['SECRET_KEY'] = os.urandom(32).hex()  # Change to a strong secret
app.config['CACHE_TYPE'] = 'simple'  # Simple in-memory cache
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout (in seconds)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.StrictRedis(host='redis-server', port=6379)

cache = Cache(app)
executor = Executor(app)

DOWNLOAD_FOLDER = os.path.abspath("./local")  # or wherever your files are stored

openai.api_key = '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'
stripe.api_key = 'sk_test_51OncNPGfeF8U30tWYUqTL51OKfcRGuQVSgu0SXoecbNiYEV70bb409fP1wrYE6QpabFvQvuUyBseQC8ZhcS17Lob003x8cr2BQ'

global nmbc
nmbc = NMCYBlockchain()
global coin
coin = Coin()
global blockchain
blockchain = Blockchain()
blockchain.create_genesis_block()
global network
network = Network()
network.create_genesis_block()
node_bc = NodeBlockchain()
PORT = random.randint(5000,6000)

app.config['CELERY_BROKER_URL'] = 'redis://red-cv8uqftumphs738vdlb0:6379'
app.config['CELERY_RESULT_BACKEND'] = 'redis://red-cv8uqftumphs738vdlb0:6379' 

app.register_blueprint(kaggle_bp, url_prefix="/app")
app.register_blueprint(sequential_bp, url_prefix="/seq")

register_template_filters(app)
# # 
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(result_backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.beat_schedule = {
    'run-every-minute': {
        'task': 'tasks.my_periodic_task',
        'schedule': 60.0,  # Run every 60 seconds
    },
}

@cache.cached(timeout=300)
def get_yf_data(ticker):
	proxy = YFinanceProxyWrapper(proxy_list)
	return proxy.fetch(ticker, period='1wk', interval='1d')

@app.route('/my_portfolio/<name>', methods=['GET'])
@login_required
def portfolio(name):
	user = current_user
	portfolio = Portfolio.query.filter_by(username=user.username,name=name).all()
	return render_template('portfolio.html', portfolio=portfolio)

@app.route("/launch-jupyter")
def launch_jupyter():
    import subprocess
    subprocess.Popen(["jupyter", "notebook"])
    return redirect("http://localhost:8888")  

@app.route('/notifications/send', methods=['GET','POST'])
@login_required
def send_notification():
	if request.method == 'POST':
		sender = request.form['sender']
		receiver_id = request.form['receiver']
		message = request.form['message']

		if not receiver_id or not message:
			return jsonify({"error": "Missing required fields"}), 400

		notification = Notification(
			sender_id=sender,
			receiver_id=receiver_id,
			message=message,
			receipt=os.urandom(10).hex()
		)
		db.session.add(notification)
		db.session.commit()
		return jsonify({"message": "Notification sent successfully"}), 201
	return render_template("send_note.html")

@app.route('/get/notes', methods=['GET','POST'])
@login_required
def get_notifications():
	u = current_user
	user = Users.query.filter_by(username=u.username).first()
	notifications = Notification.query.filter_by(receiver_id=u.username).order_by(Notification.timestamp.desc()).all()
	return render_template("my_notes.html",notes=notifications)

@app.route('/delete/notification/<receipt>', methods=['GET','POST'])
@login_required
def delete_notifications(receipt):
	notifications = Notification.query.filter_by(receipt=receipt).first()
	if notifications:
		db.session.delete(notifications)
		db.session.commit()
		return jsonify({"message": "Notification deleted successfully"}), 200
	return render_template("my_notes.html")

@app.route('/option/greeks', methods=['GET','POST'])
def calc_option_greeks():
	from greeks import black_scholes_greeks
	ls = []
	df = {"name":[],"delta": [], "gamma": [], "theta": [], "vega": [], "rho": [],'price':[],"receipt":[]}
	invests = InvestmentDatabase.query.all()
	for i in invests:
		try:
			session = requests.Session()
			session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
			t = yf.Ticker(i.investment_name.upper(),session=session)
			type_option = str(i.investment_type).strip('InvestmentType.')
			print(type_option)
			if type_option == 'u':
				OT = 'put'
				price = t.history(period='1d',interval='1m')['Close'].iloc[-1]
				greeks = black_scholes_greeks(price, i.target_price, i.time_float, i.risk_neutral, i.spread, option_type=OT)
			else:
				price = t.history(period='1d',interval='1m')['Close'].iloc[-1]
				greeks = black_scholes_greeks(price, i.target_price, i.time_float, i.risk_neutral, i.spread, option_type=type_option)
			df["name"].append(i.investment_name)
			df["delta"].append(np.round(greeks['Delta'],5))
			df["gamma"].append(np.round(greeks['Gamma'],5))
			df["theta"].append(np.round(greeks["Theta"],5))
			df["vega"].append(np.round(greeks["Vega"],5))
			df["rho"].append(np.round(greeks["Rho"],5))
			df["price"].append(np.round(greeks["Option Price"],5))
			df["receipt"].append(i.receipt)
		except Exception as e:
			print(type_option)
			print(f"Error calculating greeks for {i.investment_name}: {str(e)}")
			pass
		
	html = pd.DataFrame(df).to_html()
	string =f"""<!DOCTYPE html>
		<html lang="en">
		<head>
			<meta charset="UTF-8">
			<meta name="viewport" content="width=device-width, initial-scale=1.0">
			<title>Stochastic Greeks</title>
			<style>
				/* General page styles */
				body {{
					font-family: Arial, sans-serif;
					margin: 20px;
					background-color: #f4f4f9;
					color: #333;
				}}

				h1 a {{
					text-decoration: none;
					color: #3498db;
					font-size: 24px;
				}}

				h1 a:hover {{
					text-decoration: underline;
				}}

				/* Table styles */
				table {{
					width: 100%;
					border-collapse: collapse;
					margin-top: 20px;
					background-color: #fff;
					box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
				}}

				th, td {{
					border: 1px solid #ddd;
					padding: 12px;
					text-align: left;
				}}

				th {{
					background-color: #2c3e50;
					color: white;
					text-transform: uppercase;
					letter-spacing: 0.05em;
				}}

				tr:hover {{
					background-color: #f1f1f1;
				}}

				tr:nth-child(even) {{
					background-color: #f9f9f9;
				}}
			</style>
		</head>
		<body>
			<h1><a href="/">Back</a></h1>
			
			<!-- Insert dynamic HTML content here -->
			{html} 
		</body>
		</html>"""
	return render_template_string(string, html=html)

def recalculate():
	invests = InvestmentDatabase.query.all()
	for i in invests:
		try:
			# session = requests.Session()
			# get
			# session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
			# t = yf.Ticker(,session=session)
			prices_vector = get_yf_data(i.investment_name.upper())#t.history(period='5d',interval='1m')
			price = prices_vector['Close'].iloc[-1]
			s = stoch_price(1/52, i.time_float, i.risk_neutral, i.spread, i.reversion, price, i.target_price)
			i.stoch_price = s
			# i.tokenized_price
			db.session.commit()
		except:
			pass

def update_portfolio():
	portfolio = Portfolio.query.all()
	for i in portfolio:
		# session = requests.Session()
		# session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		t = get_yf_data(i.token_name.upper())#yf.Ticker(i.token_name.upper(),session=session)
		prices_vector = t #t.history(period='5d',interval='1m')
		price = t['Close'].iloc[-1]
		i.price = price
		db.session.commit()

def change_value_update():
	invests = InvestmentDatabase.query.all()
	for i in invests:
		proxy = YFinanceProxyWrapper(proxy_list) #requests.Session()
		# session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		# t = #yf.Ticker(i.investment_name.upper(),session=session)
		prices_vector = get_yf_data(i.investment_name.upper()) #proxy.fetch(i.investment_name.upper(),period='5d',interval='1m')#t.history(period='5d',interval='1m')
		price = prices_vector['Close'].iloc[-1]#t.history(period='1d',interval='1m')
		change = np.log(price) - np.log(i.starting_price)
		i.change_value = change
		db.session.commit()

def update_prices():
	"""
	Update investment data, including market price, time_float, tokenized price, and coins.
	"""
	print("Update")
	recalculate()
	change_value_update()
	update_portfolio()
	invests = InvestmentDatabase.query.all()

	for i in invests:
		try:
			# session = requests.Session()
			# session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
			proxy = YFinanceProxyWrapper(proxy_list)
			# Fetch current market data
			# t = #yf.Ticker(i.investment_name.upper(),session=session)
			prices_vector = proxy.fetch(i.investment_name.upper(),period='5d', interval='1m')#t.history(period='5d', interval='1m')
			price = prices_vector['Close'].iloc[-1]
			
			# Update market price
			i.market_price = price

			# Calculate time difference since the last update
			current_time = datetime.now()
			time_difference = (current_time - i.timestamp).total_seconds()  # Time elapsed in seconds

			# Update time_float (time remaining until maturity)
			i.time_float -= time_difference / (365.25 * 24 * 3600)  # Convert seconds to years

			# Update timestamp to the current time
			i.timestamp = current_time

			# Calculate tokenized price and coins
			i.tokenized_price = i.market_price / i.quantity  # Simplified for now
			i.coins = i.tokenized_price * (1 + i.spread) ** i.time_float

			# Commit changes to the database
			db.session.commit()

		except Exception as e:
			# Log the exception for debugging
			print(f"Error updating investment {i.investment_name}: {e}")
			db.session.rollback()  # Rollback in case of errors

	return 0

# schedule.every(1).minutes.do(update_prices)


@celery.task	
def update():
	"""
	Update investment data, including market price, time_float, tokenized price, and coins.
	"""
	print("Update")
	recalculate()
	change_value_update()
	update_portfolio()
	invests = InvestmentDatabase.query.all()

	for i in invests:
		try:
			# session = requests.Session()
			# session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
			# Fetch current market data
			proxy = YFinanceProxyWrapper(proxy_list)
			t = proxy.fetch(i.investment_name.upper(),period='5d', interval='1m')#yf.Ticker(i.investment_name.upper(),session=session)
			prices_vector =  t #t.history(period='5d', interval='1m')
			price = t['Close'].iloc[-1]
			
			# Update market price
			i.market_price = price

			# Calculate time difference since the last update
			current_time = datetime.now()
			time_difference = (current_time - i.timestamp).total_seconds()  # Time elapsed in seconds

			# Update time_float (time remaining until maturity)
			i.time_float -= time_difference / (365.25 * 24 * 3600)  # Convert seconds to years

			# Update timestamp to the current time
			i.timestamp = current_time

			# Calculate tokenized price and coins
			i.tokenized_price = i.market_price / i.quantity  # Simplified for now
			i.coins = i.tokenized_price * (1 + i.spread) ** i.time_float

			# Commit changes to the database
			db.session.commit()

		except Exception as e:
			# Log the exception for debugging
			print(f"Error updating investment {i.investment_name}: {e}")
			db.session.rollback()  # Rollback in case of errors

	return 0

# schedule.every(1).minutes.do(update)

@login_manager.user_loader
def load_user(user_id):
	update.delay()
	coin_db = CoinDB()
	db.session.add(coin_db)
	db.session.commit()
	betting_house = BettingHouse()
	db.session.add(betting_house)
	db.session.commit()
	return Users.query.get(int(user_id))

@app.route('/chat', methods=['GET','POST'])
def chat():
	try:
		if request.method == "POST":
			# Get the user's input from the request body
			data = request.json
			user_message = data.get('message')
			
			# Make sure the input exists
			if not user_message:
				return jsonify({"error": "No input message provided"}), 400
			
			# Call the OpenAI API to get a response from ChatGPT
			response = openai.ChatCompletion.create(
				model="gpt-4o-mini",  # You can replace with other models like gpt-4
				messages=[
					{"role": "system", "content": "You are ChatGPT, a helpful assistant."},
					{"role": "user", "content": user_message}
				]
			)
			
			# Extract the message from the API response
			chatgpt_reply = response['choices'][0]['message']['content']
			
			# Return the response as a JSON object
			return jsonify({"response": chatgpt_reply})
	
	except Exception as e:
		return jsonify({"error": str(e)}), 500
	
	return render_template('chat.html')

@app.route('/index.html')
def index():
	return render_template("index.html")

@app.route('/house')
def house():
	bet = BettingHouse.query.get_or_404(1)
	ls = {'coins':bet.coins,'cash':bet.balance}
	return jsonify(ls)

@app.route('/coin')
def coin_db():
	coin_db = CoinDB.query.get_or_404(1)
	ls = {'market_cap':coin_db.market_cap,
		 'staked_coins':coin_db.staked_coins,
		 'new_coins':coin_db.new_coins,
		 'dollar_value': coin_db.dollar_value,
		 'total_coins': coin_db.total_coins}
	return jsonify(ls)


@app.route('/buy/cash', methods=['GET'])
@login_required
def buy_cash():
	return render_template('stripe-payment.html')


@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
	try:
		user_id = current_user.id  # Assuming you're using Flask-Login
		
		checkout_session = stripe.checkout.Session.create(
			payment_method_types=['card'],
			line_items=[{
				'price_data': {
					'currency': 'usd',
					'product_data': {
						'name': 'Purchase Cash',
					},
					'unit_amount': 5000,  # Amount in cents ($50.00)
				},
				'quantity': 1,
			}],
			mode='payment',
			success_url=url_for('success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
			cancel_url=url_for('cancel', _external=True),
			metadata={
				'user_id': user_id  # Store user_id in the metadata
			}
		)
		return redirect(checkout_session.url, code=303)
	except Exception as e:
		return jsonify(error=str(e)), 403
	
@app.route('/success')
def success():
	session_id = request.args.get('session_id')
	session = stripe.checkout.Session.retrieve(session_id)
	if session.payment_status == 'paid':
		user_id = session.metadata['user_id']  # Retrieve user_id from metadata
		user = Users.query.get_or_404(user_id)
		user_balance = WalletDB.query.filter_by(address=user.username).first()
		user_balance.balance += 50  # Adding $50 to user's balance, modify as needed
		db.session.commit()
		pay_id =  session.payment_intent
		user.payment_id = pay_id
		db.session.commit()
		return f"<h1>Payment Successful</h1><a href='/'>Home</a><h3>{pay_id}</h3>"
	else:
		return '<h1>Payment Failed</h1><a href="/">Home</a>'
	
@app.route('/cancel')
def cancel():
	return '<h1>Payment Cancelled</h1><a href="/">Home</a>'

@app.route('/sell/cash', methods=['GET', 'POST'])
@login_required
def sell_cash():
	if request.method == 'POST':
		amount = request.form['amount']
		user_id = current_user.id  # Assuming you're using Flask-Login
		user = Users.query.get(user_id)
		user_balance = WalletDB.query.filter_by(address=user.username).first()
		
		if user_balance.balance >= float(amount):
			# Deduct the balance from the user's wallet
			user_balance.balance -= float(amount)
			db.session.commit()
			
			# Create a refund in Stripe
			try:
				# You need to keep track of the payment intent ID during the payment process
				payment_intent_id = request.form['payment_intent_id']  # You'll need to pass this from the frontend
				refund = stripe.Refund.create(
					payment_intent=payment_intent_id,
					amount=int(float(amount) * 100),  # amount in cents
				)
				return jsonify({'message': 'Refund Successful', 'refund': refund}), 200
			except Exception as e:
				return jsonify(error=str(e)), 403
		else:
			return jsonify({'message': 'Insufficient Balance'}), 400
	return render_template('sell-cash.html')

@app.route('/')
def base():
	return render_template('index_base.html')

@app.route('/signup', methods=['POST','GET'])
def signup():
	if request.method =="POST":
		password = request.values.get("password")
		username = request.values.get("username")
		email = request.values.get("email")
		cell_number = request.values.get("cell_number")
		unique_address = os.urandom(10).hex()
		hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
		new_user = Users(username=username, email=email,cell_number=cell_number, 
				   password=hashed_password,
				   personal_token=os.urandom(10).hex(),
				   private_token=unique_address)
		db.session.add(new_user)
		db.session.commit()
		return jsonify({'message': 'User created!'}), 201
	return render_template("signup.html")

@app.route('/signup/wallet', methods=['POST','GET'])
def create_wallet():
	if request.method =="POST":
		username = request.values.get("username")
		password = request.values.get("password")
		users = Users.query.all()
		ls = [user.username for user in users]
		passwords = [user.username for user in users]
		if username in ls:
			if password in passwords:
				data = os.urandom(10).hex()
				new_wallet = WalletDB(address=username,token=username,password=password,coinbase_wallet=data)
				db.session.add(new_wallet)
				db.session.commit()
				return jsonify({'message': 'Wallet Created!'}), 201
	return render_template("signup-wallet.html")


@app.route('/login', methods=['POST', 'GET'])
def login():
	if request.method == "POST":
		username = request.values.get("username")
		password = request.values.get("password")
		user = Users.query.filter_by(username=username).first()
		if user and bcrypt.check_password_hash(user.password, password):
			login_user(user, remember=True)  # <-- Ensure "remember=True" for session persistence
			return redirect('/')
		else:
			flash("Invalid username or password. Please try again.", "danger")
			return redirect('/login')
	return render_template("login.html")


@app.route('/get/users', methods=['GET'])
@login_required
def get_users():
	users = Users.query.all()
	users_list = [{'id': user.id, 'username': user.username,'publicKey':str(user.personal_token)} for user in users]
	return jsonify(users_list)

@app.route('/signup/val', methods=['POST','GET'])
def signup_val():
	if request.method =="POST":
		password = request.values.get("password")
		username = request.values.get("username")
		users = Users.query.all()
		ls = [user.username for user in users]
		if username in ls:
			email = request.values.get("email")
			pk = str(os.urandom(10).hex())
			hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
			new_val = Peer(user_address=username, email=email, password=hashed_password,pk=pk)
			db.session.add(new_val)
			db.session.commit()
			return jsonify({'message': 'Val created!'}), 201
		else:
			pass
	return render_template("signup-val.html")


@app.route('/investments', methods=['GET'])
def get_investments():
	update.delay()
	update_prices()
	page = request.args.get('page', 1, type=int)
	per_page = 5

	investments = InvestmentDatabase.query.paginate(page=page, per_page=per_page, error_out=False)

	return render_template(
		'investments.html',
		investments=investments,
		page=investments.page,
		total_pages=investments.pages,
		has_next=investments.has_next,
		has_prev=investments.has_prev
	)

@app.route('/get/vals')
def get_vals():
	peers = Peer.query.all()
	peers_list = [{'id': peer.id, 'username': peer.user_address,'public_key':str(peer.pk)} for peer in peers]
	return jsonify(peers_list) #render_template('validators.html', vals=validators)

@app.route('/peer/<address>/<password>', methods=['GET'])
def get_peer(address,password):
	user = Peer.query.filter_by(user_address=address).first()
	if user and bcrypt.check_password_hash(user.password, password):
		return jsonify({'id': user.id,'coins':user.miner_wallet,'cash':user.cash})
	else:
		return "Wrong Password"

@app.route('/usercred', methods=["GET","POST"])
def user_cred():
	if request.method == "POST":
		user = request.values.get("cred")
		password = request.values.get("password")
		return redirect(f'/users/{user}/{password}')
	return render_template('user-cred.html')

@app.route('/valcred', methods=["GET","POST"])
def val_cred():
	if request.method == "POST":
		user = request.values.get("username")
		password = request.values.get("password")
		peer = Peer.query.filter_by(user_address=user).first()
		ls = {'id':peer.id,'user_address':peer.user_address,'coins':peer.miner_wallet,'cash':peer.cash}
		return jsonify(ls) 
	return render_template('val-cred.html')

@app.route('/my/transactions',methods=['GET','POST'])
def my_trans():
	if request.method == "POST":
		username = request.values.get('username')
		trans = TransactionDatabase.query.filter_by(username=username).all()
		ls = [{'name':t.username,'amount':t.amount,'type':str(t.type),'from_address':t.from_address,'to_address':t.to_address,'txid':t.txid} for t in trans]
		return jsonify(ls)
	return render_template("mytans.html")

@app.route('/html/my/transactions',methods=['GET','POST'])
def my_html_trans():
	if request.method == "POST":
		username = request.values.get('username')
		trans = TransactionDatabase.query.filter_by(username=username).all()
		return render_template("view_trans.html",trans=trans)
	return render_template("mytans.html")
		
@app.route('/users/<user>/<password>', methods=['GET'])
def get_user(user,password):
	user = Users.query.filter_by(username=user).first()
	if user and bcrypt.check_password_hash(user.password, password):
		return jsonify({'id': user.id, 'username': user.username,
				   'email': user.email,
				   'private_key':str(user.private_token),
				   'personal_token':str(user.personal_token),
				   'payment_id':user.payment_id})
	else:
		return redirect('/')

@app.route('/transact',methods=['GET','POST'])
def create_transact():
	if request.method == "POST":
		id_from = request.values.get("username_from")
		id_to = request.values.get("username_to")
		value = .9*float(request.values.get("value"))
		stake = coin.process_coins()
		password = request.values.get("password")
		user = Users.query.filter_by(username=id_from).first()
		user2 = Users.query.filter_by(username=id_to).first()
		w1 = WalletDB.query.filter_by(address=id_from).first()
		w2 = WalletDB.query.filter_by(address=id_to).first()
		packet = str({'from':id_from,'to':id_to,'value':value}).encode()
		blockchain.add_transaction(packet.hex())
		pending = PendingTransactionDatabase(
									   txid=os.urandom(10).hex(),
									   username=id_from,
									   from_address=w1.address,
									   to_address=id_to,
									   amount=value,
									   timestamp=dt.datetime.now(),
									   type='internal_wallet',
									   signature=str(w1.address).encode().hex())
		db.session.add(pending)
		db.session.commit()
		from_addrs = user.username
		to_addrs = user2.username
		txid = str(os.urandom(10).hex())
		transaction = {
				 'index': len(blockchain.pending_transactions)+1,
				 'previous_hash': sha512(str(blockchain.get_latest_block()).encode()).hexdigest(),
				 'timestamp':dt.date.today(),
				 'transactions': blockchain.pending_transactions,
				 'hash':sha256(str(blockchain.pending_transactions).encode())}
		blockchain.receipts['to'] = user2.username
		blockchain.receipts['from'] = user.username
		blockchain.receipts['value'] = value
		blockchain.receipts['txid'] = txid
		network.add_transaction(blockchain.pending_transactions)
		blockchain.add_transaction(transaction)
		blockchain.money.append(value)
		if user and bcrypt.check_password_hash(user.password, password):
			betting_house = BettingHouse.query.get_or_404(1)
			betting_house.cash_fee(.1*value)			
			new_value = 0.9*value
			w1.set_transaction(w2, new_value)
			new_transaction = TransactionDatabase(
										 username=user.username,
										 txid=txid,
										 from_address = from_addrs,
										 signature=os.urandom(10).hex(),
										 to_address = to_addrs,
										 amount = value, 
										 type='send')
			db.session.add(new_transaction)
			db.session.commit()
			coin_db = CoinDB.query.get_or_404(1)
			coin_db.gas(blockchain,6)
			return  """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""
	return render_template("trans.html")

@app.route('/liquidate', methods=["POST","GET"])
def delete_asset():
	if request.method == 'POST':
		address = request.values.get('address')
		user = request.values.get('user')
		password = request.values.get('password')
		user_db = Users.query.filter_by(username=user).first()
		wal = WalletDB.query.filter_by(address=user).first()
		transaction = TransactionDatabase.query.filter_by(username=user).first()
		asset = InvestmentDatabase.query.filter_by(receipt=address).first()
		if asset.investors == 1 and password == asset.password and user == asset.owner:
			wal.coins += asset.coins_value
			db.session.commit()
			db.session.delete(asset)
			db.session.commit()
			return "<h1>Asset Closed</h1>"
		else:
			return "<h1>Can't close position</h1>"
	return render_template("close_asset.html")

@app.route('/get/tokens', methods=["POST","GET"])
def get_asset_token():
	asset_tokens = AssetToken.query.all()
	ls = [{'id':asset.id,
		'token_address':asset.token_address,
		'user_address':asset.user_address,
		'transaction_receipt':asset.transaction_receipt,
		'username':asset.username,
		'coins':asset.coins} for asset in asset_tokens]
	return jsonify(ls)

@app.route('/make/block')
def make_block():
	if not blockchain:
		return "<h3>Blockchain instance not found</h3>"
	index = len(blockchain.chain) + 1
	previous_block = blockchain.get_latest_block()
	previous_hash = (str(blockchain.get_latest_block()).encode().hex())#if previous_block else '0'
	timestamp = dt.date.today()
	transactions = blockchain.pending_transactions
	index = len(network.chain) + 1
	previous_block = network.get_latest_block()
	previous_hash = (str(network.get_latest_block()).encode()).hex()#if previous_block else '0'
	timestamp = dt.datetime.now()
	transactions = blockchain.pending_transactions
	# Create a new block
	block_data = {
		'index': index,
		'previous_hash': previous_hash,
		'timestamp': timestamp,
		'transactions': transactions,
	}
	
	block_string = str(block_data).encode()
	block_hash = hashlib.sha256(block_string).hexdigest()
	block = PrivateBlock(index, 
					  previous_hash, 
					  timestamp,
					  str(transactions),
					  block_hash)
	new_block = Block(
		index=int(index),
		previous_hash=str(previous_hash),
		timestamp=timestamp,
		transactions=str(transactions).encode(),
		hash=str(block_hash)
	)
	db.session.add(new_block)
	db.session.commit()
	blockchain.add_block(block)
	blockchain.approved_transactions.append(transactions)
	coin_db = CoinDB.query.get_or_404(1)
	coin_db.gas(blockchain,4)
	staked_coins = coin_db.proccess_coins(blockchain)
	t = blockchain.staked_coins.append(staked_coins)
	return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3>{str(new_block).encode().decode()}"""

@app.route('/cmc',methods=['GET'])
def cmc():
	user = current_user
	return render_template("cmc.html",user=user)

@login_required
@app.route('/html/trans',methods=['GET'])
def html_trans_database():
	t = TransactionDatabase.query.all()
	return render_template("html-trans.html", trans=t)

@app.route('/mine/investments',methods=['GET','POST'])
def mine_investments():
	update.delay()
	update_prices()
	return """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""


@login_required
@app.route('/html/investment/ledger',methods=['GET'])
def html_investment_ledger():
	update.delay()
	update_prices()
	t = InvestmentDatabase.query.all()
	return render_template("html-invest-ledger.html", invs=t)

@app.route('/get/approved',methods=['GET'])
def get_approved():
	trans =  Chain.query.all()
	ls = [{'id':t.id,'txid':t.txid,'username':t.username,'from':t.from_address,
		'to':t.to_address,'amount':t.amount,'time':t.timestamp,'type':str(t.type),'signature':t.signature} for t in trans]
	return jsonify(ls)

@app.route('/get/blocks',methods=['GET'])
def get_blocks():
	transports = Block.query.all()
	transports_list = [{'id': t.id,'index':str(t.index),'transactions':str(t.transactions)} for t in transports]
	return jsonify(transports_list)

@app.route('/get/block/<int:id>',methods=['GET'])
@login_required
def html_block(id):
	block = Block.query.get_or_404(id)
	return jsonify({'transactions':str(block.transactions),'id':block.id})

@app.route('/html/block/<int:id>',methods=['GET'])
@login_required
def get_block(id):
	block = Block.query.get_or_404(id)
	return render_template("html-block.html",block=block)

@app.route('/add/portfolio',methods=['POST','GET'])
@login_required
def add_portfolio():
	if request.method == 'POST':
		ticker = request.form.get('name').upper()
		password = request.form.get('password')
		name = request.form.get('pname').lower()
		session = requests.Session()
		session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		price = yf.Ticker(ticker,session=session).history(period='ytd',interval='1d')['Close']
		mean = np.mean(price)
		std = np.std(price)
		weight = float(request.form.get('weight'))
		user = current_user
		prt = Portfolio(name=name,mean=mean,std=std,weight=weight,price=price[-1],username=user.username,
				token_name=ticker.upper(),token_address=os.urandom(10),user_address=user.personal_token,transaction_receipt=os.urandom(10))
		db.session.add(prt)
		db.session.commit()
		return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3>Portfolio {ticker}"""
	return render_template("add_port.html")


@app.route('/holdings', methods=['GET'])
@login_required
def get_user_wallet():
	update.delay()
	update_prices()
	user = current_user
	
	# Fetch the user's wallet
	wallet = WalletDB.query.filter_by(address=user.username).first()
	
	if not wallet:
		return "Wallet not found", 404
	
	# Fetch all asset tokens for the user
	assets = AssetToken.query.filter_by(username=user.username).all()
	
	# Initialize the dataframe dictionary
	df = {
		'inv_name': [], 
		'quantity': [], 
		'marketcap': [], 
		'starting_price': [], 
		'market_price': [], 
		'coins_value': [], 
		'change_value': [],
		'tokenized_price':[],
		'stochastic_price':[],
		'receipt':[],
	}
	
	total_investments = 0
	profit_loss = 0
	for asset in assets:
		invs = InvestmentDatabase.query.filter_by(receipt=asset.transaction_receipt).first()
		if invs:
			df['inv_name'].append(invs.investment_name)
			df['quantity'].append(asset.quantity)
			df['marketcap'].append(invs.market_cap)
			df['starting_price'].append(invs.starting_price)
			df['market_price'].append(invs.market_price)
			df['coins_value'].append(invs.coins_value)
			df['change_value'].append(invs.change_value)
			df['tokenized_price'].append(invs.tokenized_price)
			df['stochastic_price'].append(invs.stoch_price)
			df['receipt'].append(invs.receipt)
			profit_loss+=invs.change_value
			total_investments+=invs.tokenized_price*asset.quantity
    
    # Convert the dictionary to a pandas DataFrame
	dataframe = pd.DataFrame(df)
		
    # Transport list for potential future JSON response
	transports_list = [{"address": wallet.address, "balance": wallet.balance, "coins": wallet.coins}]
    
		# Convert DataFrame to HTML table with styles
	html = dataframe.to_html(index=False)
	html_table_with_styles = f"""
	<title>Holdings</title>
	<meta name="viewport" content="width=device-width, initial-scale=1.0">

		<style>
			table {{
				width: 100%;
				border-collapse: collapse;
			}}
			th, td {{
				border: 1px solid black;
				padding: 10px;
				text-align: left;
			}}
			th {{
				background-color: #f2f2f2;
			}}
			tr:nth-child(even) {{
				background-color: #f9f9f9;
			}}
		</style>
		<h1><a href="/">Back</a></h1>
		{html}
		<table>
				<thead>
					<tr>
						<th>Profit Loss (%)</th>
						<th>Total Investments</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>{profit_loss}</td>
						<td>{total_investments}</td>
					</tr>
				</tbody>
			</table>
	"""
	
	# Render the HTML table as a response
	return render_template_string(html_table_with_styles)

@app.route('/html/mywallet',methods=['GET'])
@login_required
def html_wallet():
	user = current_user
	wallet = WalletDB.query.filter_by(address=user.username).first()
	return render_template("wallet.html",wallet=wallet)


@app.route('/bc/receipts',methods=['GET'])
@login_required
def get_bc_receipts():
	df = pd.DataFrame(blockchain.receipts)
	return df.to_html()


@app.route('/get/trans/<int:id>',methods=['GET'])
@login_required
def get_transaction(id):
	t = TransactionDatabase.query.get_or_404(id)
	transports_list = [{'user':t.username,'id': t.id, 'from_address':str(t.from_address),'to_address':str(t.to_address),'value':t.amount,'txid':t.txid}]
	return jsonify(transports_list)


@app.route('/get/block/<int:id>',methods=['GET'])
@login_required
def get_block_id(id):
	t = TransactionDatabase.query.get_or_404(id)
	transports_list = [{'user':t.username,'id': t.id, 'from_address':str(t.from_address),'to_address':str(t.to_address),'value':t.amount,'txid':t.txid}]
	return jsonify(transports_list)


@app.route('/get/ledger',methods=['GET'])
@login_required
def get_ledger():
	trans = TransactionDatabase.query.all()
	transports_list = [{'user':t.username,'id': t.id,'type':str(t.type), 'from_address':str(t.from_address),'to_address':str(t.to_address),'value':t.amount,'txid':t.txid} for t in trans]
	return jsonify(transports_list)


@app.route('/get/wallets',methods=['GET'])
@login_required
def get_wallets():
	transports = WalletDB.query.all()
	transports_list = [{'address':t.address,'id':t.id,'user':str(t.token)} for t in transports]
	return jsonify(transports_list)

@app.route("/validate/hash",methods=['GET',"POST"])
def validate():
	if request.method == "POST":
		plain = request.values.get("plain")
		hash_value = request.values.get("hash")
		if hash_value == sha512(str(plain).encode()).hexdigest():
			return f"<h1>Valid ID</h1><h2>{plain}</h2><h2>{hash_value}</h2>"
		else:
			return "<h1>Incorrect ID</h1>"
	return render_template("validate-hash.html")


@app.route('/get/pending')
def get_pending():
	trans = PendingTransactionDatabase.query.all()
	ls = [{'id':t.id,'txid':t.txid,'username':t.username,'from':t.from_address,
		'to':t.to_address,'amount':t.amount,'time':t.timestamp,'type':str(t.type),'signature':t.signature} for t in trans]
	return jsonify(ls)


@app.route('/mine', methods=['GET', 'POST'])
def mine():
	if request.method == 'POST':
		blockdata = Block.query.all()
		user_address = request.values.get("user_address")
		miner = Peer.query.filter_by(user_address=user_address).first()
		n = network.get_stake()
		staked_coins = [10] # Initialize with the first stake value as an integer
		coin_db = CoinDB.query.get_or_404(1)
		for i in blockdata:
			status = blockchain.is_chain_valid()
			s_status = network.is_chain_valid()
			print('\nthe status is\n', status)
			print('\nthe status is\n', s_status)
			pending_transactions = PendingTransactionDatabase.query.all()
			for i in pending_transactions:
				approved_transaction = Chain(txid=i.txid,
								  username=i.username,
								  from_address=i.from_address,
								  to_address=i.to_address,
								  amount=i.amount,
								  timestamp=i.timestamp,
								  type=i.type,
								  signature=i.signature)
				db.session.add(approved_transaction)
				db.session.commit()
			nonce, hash_result, time_taken = blockchain.proof_of_work(i, 5)
			nonce, hash_result, time_taken = network.proof_of_work(i, 5)
			staked_proccess = coin.process_coins()
			coin_db.gas(blockchain,10)
			all_approved_transactions = Chain.query.all()
			approved_values = [i.amount for i in all_approved_transactions]
			amount_values = [i.amount for i in pending_transactions]
			print(amount_values)
			stake = coin.stake_coins(approved_values,amount_values)
			coin_db.staked_coins+=stake
			db.session.commit()
			blockchain.market_cap += stake 
			staked_coins.append(stake)
			staked_coins.append(coin_db.new_coins)
			blockchain.mine_pending_transactions(1)
			value = sum(staked_coins)/len(staked_coins)
			for i in pending_transactions:
				db.session.delete(i)
				db.session.commit()
			staked_coins = []
		miner.miner_wallet+=value
		db.session.commit()
		packet = {
			'index': len(blockchain.chain) + 1,
			'previous_hash': sha256(str(blockchain.get_latest_block()).encode()).hexdigest(),
			'datetime': str(dt.datetime.now()),
			'transaction': 'mining_complete' ,
		}
		encoded_packet = str(packet).encode().hex()
		blockchain.add_block(encoded_packet)
		node_bc.broadcast_block(packet)
		return f"<h1><a href='/'> Home </a></h1><h3>Success</h3>You've mined {value} coins"
	return render_template('mine.html')


# proxy_create_investment = YFinanceProxyWrapper(proxy_list)
@app.route('/create/investment', methods=['GET', 'POST'])
def buy_or_sell():
    def normal_pdf(x, mean=0, std_dev=1):
        return scipy.stats.norm.pdf(x, loc=mean, scale=std_dev)

    def C(s):
        K = s ** 3 - 3 * s ** 2 * (1 - s)
        return (s * (s - 1 / K)) ** (1 / s) * normal_pdf(s)
		
    if request.method == "POST":
        try:
            user = request.values.get('name')
            invest_name = request.values.get('ticker')
            coins = request.values.get('coins')
            password = request.values.get('password')
            qt = request.values.get("qt")
            target_price = request.values.get("target_price")
            maturity = request.values.get("maturity")
            risk_neutral = request.values.get("eta")
            spread = request.values.get("sigma")
            reversion = request.values.get("mu")
            option_type = request.values.get("option_type")

            if not all([user, invest_name, coins, password, qt, target_price, maturity, risk_neutral, spread, reversion, option_type]):
                return "<h3>Missing required fields</h3>"

            # invest_name = invest_name.upper()
            coins, qt, target_price, maturity = map(float, [coins, qt, target_price, maturity])
            risk_neutral, spread, reversion = map(float, [risk_neutral, spread, reversion])
            option_type = option_type.lower()
            user_db = Users.query.filter_by(username=user).first()
            if not user_db:
                return "<h3>User not found</h3>"
            history = yf.Ticker(invest_name).history()#proxy_create_investment.fetch(invest_name,period='1d', interval='1m') 
			# print(history)
            if history.empty:
                return "<h3>Invalid ticker symbol</h3>"
	
            price = history['Close'].iloc[-1]
            sigma = np.std(history['Close'].pct_change().dropna()) * np.sqrt(525960)
            option = black_scholes(price, target_price, maturity, 0.05, sigma, option_type)

            stoch = stoch_price(0.003968253968, maturity, risk_neutral, spread, reversion, price, target_price, option_type)
            token_price = max(0, option + derivative_price(history['Close'], risk_neutral, reversion, spread)) + C(coins)

            wal = WalletDB.query.filter_by(address=user).first()
            if wal and wal.coins >= coins:
                receipt = os.urandom(10).hex()

                new_transaction = TransactionDatabase(
                    txid=receipt,
                    from_address=user_db.personal_token,
                    to_address=invest_name,
                    amount=coins * qt,
                    type='investment',
                    username=user,
                    signature=sha256(str(user_db.private_token).encode()).hexdigest()
                )
                db.session.add(new_transaction)
                db.session.commit()

                new_asset_token = AssetToken(
                    username=user,
                    token_address=receipt,
                    user_address=user_db.personal_token,
                    token_name=invest_name,
                    transaction_receipt=os.urandom(10).hex(),
                    quantity=qt,
                    cash=qt * token_price,
                    coins=coins
                )
                db.session.add(new_asset_token)
                db.session.commit()

                new_investment = InvestmentDatabase(
                    owner=user,
                    investment_name=invest_name,
                    password=password,
                    quantity=qt,
                    risk_neutral=risk_neutral,
                    spread=spread,
                    reversion=reversion,
                    market_cap=qt * price,
                    target_price=target_price,
                    investment_type=option_type,
                    starting_price=price,
                    market_price=price,
                    timestamp=dt.datetime.utcnow(),
                    time_float=maturity,
                    stoch_price=stoch,
                    coins_value=coins,
                    tokenized_price=token_price,
                    investors=1,
                    receipt=receipt
                )
                db.session.add(new_investment)
                db.session.commit()

                wal.coins -= coins
                db.session.commit()

                pen_trans = PendingTransactionDatabase.query.order_by(PendingTransactionDatabase.id.desc()).first()
                all_pending = PendingTransactionDatabase.query.all()

                new_transaction = PendingTransactionDatabase(
                    txid=os.urandom(10).hex(),
                    username=user,
                    from_address=user,
                    to_address='market',
                    amount=token_price,
                    timestamp=dt.datetime.utcnow(),
                    type='investment',
                    signature=receipt
                )
                db.session.add(new_transaction)
                db.session.commit()

                packet = {
                    'index': len(Block.query.all()) + 1,
                    'previous_hash': sha256(str(pen_trans.signature).encode()).hexdigest() if pen_trans else '0',
                    'datetime': str(dt.datetime.utcnow()),
                    'transactions': [str(tx) for tx in all_pending],
                }
                encoded_packet = str(packet).encode().hex()

                blockdata = Block(
                    index=len(Block.query.all()) + 1,
                    previous_hash=pen_trans.signature if pen_trans else '0',
                    timestamp=dt.datetime.utcnow(),
                    hash=encoded_packet,
                    transactions=str(all_pending)
                )
                db.session.add(blockdata)
                db.session.commit()

                return """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""
            else:
                return "<h3>Insufficient coins in wallet</h3>"

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3>"

    return render_template('make-investment-page.html')


@app.route('/track/inv', methods=['GET','POST'])
def track_invest():
	if request.method == 'POST':
		receipt = request.form.get('receipt')
		tracked = TrackInvestors.query.filter_by(receipt=receipt).all()
		ls = [{'id': t.id, 'receipt':t.receipt,
         'investor_name': t.investor_name, 
         'token': t.investor_token, 
         'investment_name': t.investment_name, 
         'owner': t.owner,
         'tokenized_price': t.tokenized_price} for t in tracked]
		return jsonify(ls)
	return render_template("inv-inv.html")

@app.route('/download/chain-db')
def download_chain_db():
	# Query data from the SQLAlchemy model
	data = Chain.query.all() #YourTableModel.query.all()
	
	# Create an in-memory CSV file
	def generate():
		# Write header
		yield ','.join(['ID', 'txid', 'username','from_address','to_address','amount',
			'timestamp','type','signature']) + '\n'  # Replace with actual column names
		
		# Write data rows
		for row in data:
			yield ','.join([str(row.id), str(row.txid), str(row.username),
				str(row.from_address), str(row.to_address), str(row.amount),
				str(row.timestamp), str(row.type), str(row.signature)]) + '\n'  # Replace with actual columns
			
	# Create a response object for downloading
	response = Response(generate(), mimetype='text/csv')
	response.headers.set('Content-Disposition', 'attachment', filename='chain_data.csv')
	return response


@app.route('/download/valuation-db')
def download_valuation_db():
	# Query data from the SQLAlchemy model
	data = ValuationDatabase.query.all() #YourTableModel.query.all()
	# Create an in-memory CSV file
	def generate():
		# Write header
		yield ','.join(['ID', 'owner', 'target_company','forecast','wacc','roe',
			'rd','receipt']) + '\n'  # Replace with actual column names
		
		# Write data rows
		for row in data:
			yield ','.join([str(row.id), str(row.owner), str(row.target_company),
				str(row.forecast), str(row.wacc), str(row.roe),
				str(row.rd), str(row.receipt)]) + '\n'  # Replace with actual columns
			
	# Create a response object for downloading
	response = Response(generate(), mimetype='text/csv')
	response.headers.set('Content-Disposition', 'attachment', filename='valuation_data.csv')
	return response

@app.route('/download/investment-db')
def download_investment_db():
	# Query data from the SQLAlchemy model
	data = InvestmentDatabase.query.all() #YourTableModel.query.all()
	
	# Create an in-memory CSV file
	def generate():
		# Write header
		yield ','.join(['ID', 'owner','investment_name', 'quantity','market_cap','change_value','starting_price',
			'market_price','coins_value','investors','receipt','tokenized_price']) + '\n'  # Replace with actual column names
		# Write data rows
		for row in data:
			yield ','.join([str(row.id), str(row.owner), str(row.investment_name),
				str(row.quantity), str(row.market_cap), str(row.change_value),
				str(row.starting_price), str(row.market_price), str(row.coins_value),
				str(row.investors), str(row.receipt), str(row.tokenized_price)]) + '\n'  # Replace with actual columns
	# Create a response object for downloading
	response = Response(generate(), mimetype='text/csv')
	response.headers.set('Content-Disposition', 'attachment', filename='investment_data.csv')
	return response

@app.route('/search/<receipt>')
def search(receipt):
    asset = InvestmentDatabase.query.filter_by(receipt=receipt).first()
    return render_template('search.html',asset=asset)


@app.route('/invest',methods=['GET'])
def invest_double_check_get():
	return render_template("invest.html")


@app.route('/invest',methods=['POST'])
def invest_double_check_post():
	user = request.values.get('name')
	receipt = request.values.get('address')
	staked_coins = float(request.values.get('amount'))
	password = request.values.get('password')
	user_name = Users.query.filter_by(username=user).first()
	inv = InvestmentDatabase.query.filter_by(receipt=receipt).first()
	wal = WalletDB.query.filter_by(address=user_name.username).first()
	owner_wallet = WalletDB.query.filter_by(address=inv.owner).first()
	if password == wal.password:
		if inv.quantity >= 0:
			if wal.coins >= staked_coins:
				inv.quantity -= staked_coins
				total_value = inv.tokenized_price*staked_coins
				owner_wallet.coins += 0.1 * total_value
				new_value = 0.8 * total_value
				wal.coins -= total_value
				inv.coins_value += new_value
				# Potentially Problomatic Code 
				a_tk = AssetToken(
						username=user,
						token_name=inv.investment_name,
						token_address=os.urandom(10).hex(),
						user_address=user_name.personal_token,
						transaction_receipt=inv.receipt,
						quantity = staked_coins,
						cash = coin.dollar_value*inv.tokenized_price,
						coins = inv.tokenized_price)
				db.session.add(a_tk)
				db.session.commit()
				inv.add_investor()
				return jsonify({"message": "Investment successful"}), 200
			else:
				return jsonify({"message": "Insufficient coins"}), 400
		else:
			return jsonify({"message": "Insufficient quantity of investment"}), 400
	else:
		return jsonify({"message": "Invalid password"}), 400
	return 200



@app.route('/invest/asset',methods=['GET','POST'])
@login_required
def invest():
	update.delay()
	if request.method =="POST":
		user = request.values.get('name')
		receipt = request.values.get('address')
		staked_coins = float(request.values.get('amount'))
		password = request.values.get('password')
		user_name = Users.query.filter_by(username=user).first()
		inv = InvestmentDatabase.query.filter_by(receipt=receipt).first()
		wal = WalletDB.query.filter_by(address=user_name.username).first()
		owner_wallet = WalletDB.query.filter_by(address=inv.owner).first()
		if password == wal.password:
			if inv.quantity >= 0:
				if wal.coins >= staked_coins:
					inv.quantity -= staked_coins
					db.session.commit()
					total_value = inv.tokenized_price*staked_coins
					house = BettingHouse.query.get_or_404(1)
					house.coin_fee(0.1*total_value)
					owner_wallet.coins += 0.1*total_value
					db.session.commit()
					new_value = 0.8*total_value
					wal.coins -= total_value
					inv.coins_value += new_value
					db.session.commit()
					new_transaction = TransactionDatabase(
											username=user,
											txid=inv.receipt,
											from_address=user_name.personal_token,
											to_address=inv.investment_name,
											amount=new_value,
											type='investment',
											signature=os.urandom(10).hex())
					db.session.add(new_transaction)
					db.session.commit()
					inv.add_investor()
					inv.append_investor_token(
								name=user, 
								address=user_name.personal_token, 
								receipt=inv.receipt,
								amount=staked_coins,
								currency='coins')
					a_tk = AssetToken(
						username=user,
						token_name=inv.investment_name,
						token_address=os.urandom(10).hex(),
						user_address=user_name.personal_token,
						transaction_receipt=inv.receipt,
						quantity = staked_coins,
						cash = coin.dollar_value*inv.tokenized_price,
						coins = inv.tokenized_price)
					db.session.add(a_tk)
					db.session.commit()
					track = TrackInvestors(
							receipt=receipt,
							tokenized_price=inv.tokenized_price,
							owner = sha512(str(inv.owner).encode()).hexdigest(),
							investment_name=inv.investment_name,
							investor_name=sha512(str(user_name.username).encode()).hexdigest(),
							investor_token=user_name.personal_token)
					db.session.add(track)
					db.session.commit()
					blockchain.add_transaction({
									'index':len(blockchain.chain)+1,
									"previous_hash":str(blockchain.get_latest_block()).encode().hex(),
									'timestamp':str(dt.date.today()),
									'data':str({'receipt':receipt,
												'tokenized_price':inv.tokenized_price,
												'owner':inv.owner,
												'investment_name':inv.investment_name,
												'investor_name':user_name.username,
												'investor_token':user_name.personal_token})})
					return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3><p>You've successfully invested {new_value} in {inv.investment_name}"""
				else:
					return "<h3>Insufficient coins in wallet</h3>"
	return render_template("invest-in-asset.html")

@app.route('/profile')
@login_required
def profile():
	user = current_user
	notifications = Notification.query.filter_by(receiver_id=user.username).order_by(Notification.timestamp.desc()).all()
	wallet = WalletDB.query.filter_by(address=user.username).first()
	portfolio = Portfolio.query.filter_by(username=user.username).all()
	investments = InvestmentDatabase.query.filter_by(owner=user.username).all()
	return render_template("nmbc_profile.html",user=user.username.upper(),notifications=notifications,wallet=wallet,portfolio=portfolio,investments=investments)


@app.route('/info/<int:id>')
def info(id):
	update.delay()
	asset = InvestmentDatabase.query.get_or_404(id)
	name = asset.investment_name.upper()
	url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey=6ZGEV2QOT0TMHMPZ'.format(ticker=name)
	r = requests.get(url)
	data = r.json()

	mk = data['MarketCapitalization']
	beta = data['Beta']
	DividendYield = data['DividendYield']
	industry = data['Sector']
	website = data['OfficialSite']
	description = data['Description']
	PEG = data['PEGRatio']
	PE = data['PERatio']
	session = requests.Session()
	session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
# data = yf.Ticker("AAPL", session=session)
	# res = asset_info(name)
	df = yf.Ticker(name,session=session).history(period='2y', interval='1d')["Close"]
	df = df.dropna()
	# Compute rolling mean and standard deviation
	rolling_window = 20  # Adjust the window size as needed
	df_mean = df.rolling(window=rolling_window).mean()
	df_std = df.rolling(window=rolling_window).std()
	# Compute 95% confidence intervals
	upper_bound = df_mean + 1.96 * df_std
	lower_bound = df_mean - 1.96 * df_std
	# Create figure
	fig = go.Figure()
	# Add stock price trend
	fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines', name=f"{name} Close Price", line=dict(color="blue")))
	# Add confidence interval as a shaded area
	fig.add_trace(go.Scatter(x=df.index, y=upper_bound, fill=None, mode='lines', line=dict(color='lightblue'), name="Upper Bound"))
	fig.add_trace(go.Scatter(x=df.index, y=lower_bound, fill='tonexty', mode='lines', line=dict(color='lightblue'), name="Lower Bound", opacity=0.3))

	# Set title and labels
	fig.update_layout(title=f"{name} Stock Price Trends with Confidence Interval",
					xaxis_title="Date", yaxis_title="Stock Price",
					template="plotly_dark")

	# Convert plot to JSON
	graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template("info.html", asset=asset, graph_json=graph_json, mk=mk,
						beta=beta,industry=industry,website=website,description=description,
						div=DividendYield,PEG=PEG,PE=PE,)


@app.route('/asset/info/<int:id>')
def info_assets(id):
	update.delay()
	asset = InvestmentDatabase.query.get_or_404(id)
	name = asset.investment_name.upper()
	session = requests.Session()
	session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
# data = yf.Ticker("AAPL", session=session)
	# res = asset_info(name)
	df = yf.Ticker(name,session=session).history(period='2y', interval='1d')["Close"]
	df = df.dropna()
	# Compute rolling mean and standard deviation
	rolling_window = 20  # Adjust the window size as needed
	df_mean = df.rolling(window=rolling_window).mean()
	df_std = df.rolling(window=rolling_window).std()
	# Compute 95% confidence intervals
	upper_bound = df_mean + 1.96 * df_std
	lower_bound = df_mean - 1.96 * df_std
	# Create figure
	fig = go.Figure()
	# Add stock price trend
	fig.add_trace(go.Scatter(x=df.index, y=df, mode='lines', name=f"{name} Close Price", line=dict(color="blue")))
	# Add confidence interval as a shaded area
	fig.add_trace(go.Scatter(x=df.index, y=upper_bound, fill=None, mode='lines', line=dict(color='lightblue'), name="Upper Bound"))
	fig.add_trace(go.Scatter(x=df.index, y=lower_bound, fill='tonexty', mode='lines', line=dict(color='lightblue'), name="Lower Bound", opacity=0.3))

	# Set title and labels
	fig.update_layout(title=f"{name} Stock Price Trends with Confidence Interval",
					xaxis_title="Date", yaxis_title="Stock Price",
					template="plotly_dark")

	# Convert plot to JSON
	graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template("asset-info.html", asset=asset, graph_json=graph_json)


@app.route('/get/asset/<int:id>',methods=['GET','POST'])
def get_asset(id):
	try:
		t = InvestmentDatabase.query.get_or_404(id)
		info = {'id': t.id,'name': str(t.investment_name),'owner':t.owner,'investors_num':t.investors,'market_cap':str(t.market_cap),'coins_value':str(t.coins_value),'receipt':str(t.receipt),'tokenized_price':str(t.tokenized_price),'market_price':t.market_price,'change':t.change_value,'original_price':t.starting_price}
		return jsonify(info)
	except:
		return "<h2>The asset is no longer active<h2>"

@app.route('/buy/coins',methods=['GET','POST'])
def buy_coins():
	if request.method =="POST":
		exchange = 100
		value = float(request.values.get('value'))
		id = request.values.get('id')
		username = request.values.get('username')
		password = request.values.get('password')
		house = BettingHouse.query.get_or_404(1)
		user = Users.query.filter_by(username=username).first()
		wal = WalletDB.query.filter_by(address=username).first()
		if user and bcrypt.check_password_hash(user.password, password):
			coins = float(value*exchange)
			if coins <= house.coins:
				house.coins -= coins
				db.session.commit()
				wal.balance -= value
				db.session.commit()
				wal.coins += coins
				db.session.commit()
		return """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""
	return render_template("buycash.html")

@app.route('/sell/coins',methods=['GET','POST'])
def sell_coins():
	if request.method =="POST":
		exchange = coin.dollar_value
		value = float(request.values.get('value'))
		username = request.values.get('username')
		password = request.values.get('password')
		house = BettingHouse.query.get_or_404(1)
		user = Users.query.filter_by(username=username).first()
		wal = WalletDB.query.filter_by(address=username).first()
		if user and bcrypt.check_password_hash(user.password, password):
			if wal.coins >= value:
				house.coins += .05*value
				db.session.commit()
				cash = float(value*exchange*.95)
				wal.balance += cash
				db.session.commit()
				wal.coins -= value
				db.session.commit()
		return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3><p>You've successfully sold {value} coins.</p>"""
	return render_template("sell.html")

@app.route('/my/assets',methods=['GET','POST'])
def my_assets():
	if request.method == 'POST':
		user_address = request.values.get("address")
		asset_tokens = AssetToken.query.filter_by(username=user_address).all()
		ls = [{
			'id':asset.id,
			'username':asset.username,
			'token_name':asset.token_name,
			'token_address':asset.token_address,
			'user_address':asset.user_address,
		 	'transaction_receipt':asset.transaction_receipt,
			'quantity':asset.quantity,
			'coins':asset.coins,
		 	'cash':asset.cash} for asset in asset_tokens]
		return jsonify(ls)
	return render_template("myassets.html")

@app.route("/active/assets")
def asset_options():
	update.delay()
	options = InvestmentDatabase.query.all()
	return render_template("asset-options.html",invs=options)

@app.route('/html/my/assets',methods=['GET','POST'])
def html_my_assets():
	if request.method == 'POST':
		user_address = request.values.get("address")
		asset_tokens = AssetToken.query.filter_by(username=user_address).all()
		ls = [{
		 'token_address':asset.token_address,
		 'transaction_receipt':asset.transaction_receipt,
		 'coins':asset.coins,
		 'cash':asset.cash} for asset in asset_tokens]
		return render_template("myassets-view.html",assets=asset_tokens)
	return render_template("myassets.html")
	
@app.route('/sell/asset',methods=['GET','POST'])
def sell_asset():
	update.delay()
	if request.method =="POST":
		update.delay()
		address = request.values.get('address')
		user = request.values.get('user')
		password = request.values.get('password')
		invest = InvestmentDatabase.query.filter_by(receipt=address).first()
		wal = WalletDB.query.filter_by(address=user).first()
		user_db = Users.query.filter_by(username=user).first()
		user_token = user_db.personal_token 
		asset_token = AssetToken.query.filter_by(transaction_receipt = address).first()
		if (asset_token != None) and (invest.investors > 1) and (invest != None):
				update.delay()
				close_position = ((1+invest.change_value)*asset_token.quantity)*invest.tokenized_price
				wal.coins += close_position
				invest.investors -= 1
				invest.coins_value -= close_position
				db.session.commit()
				invest.update_token_value()
				bc_trans = {
        		'receipt':asset_token.token_address,
                'from_address':'market',
                'to_address':user,
                'amount':close_position,
                'type':"liquidation"}
				blockchain.add_transaction(bc_trans)
				new_transaction = TransactionDatabase(username=user,
                                          txid=asset_token.token_address,
                                          from_address='market',
                                          to_address=user,
                                          amount=close_position,
                                          type="liquidation",
                                          signature=asset_token.transaction_receipt)
				db.session.add(new_transaction)
				db.session.commit()
				db.session.delete(asset_token)
				db.session.commit()
				return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3><h3>You have earned...{close_position}</h3>"""
		else:
			return f"""<h1>Liquidation Not Possible</h1>"""
	return render_template("liquidate.html")

@app.route('/valuation/statistics', methods=['GET','POST'])
def valuation_stats():
	if request.method=="POST":
		name = request.values.get('ticker')
		session = requests.Session()
		session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		session.proxies.update(proxy_list)
		t = yf.Ticker(name.upper(),session=session)
		price = t.history()["Close"][-1]
		invs = ValuationDatabase.query.filter_by(target_company=name).all()
		ls = {'coe':[],"cod":[],"price":[],'wacc':[],'change':[]}
		for i in invs:
			ls['coe'].append(i.roe)
			ls['cod'].append(i.rd)
			ls['wacc'].append(i.wacc)
			ls['price'].append(i.forecast)
			ls['change'].append(i.change_value)
		
		mu_coe = np.mean(ls["coe"])
		s_coe = np.std(ls["coe"])
		mu_cod = np.mean(ls["cod"])
		s_cod = np.std(ls["cod"])
		mu_wacc = np.mean(ls["wacc"])
		s_wacc = np.std(ls["wacc"])
		mu_forecast = np.mean(ls["price"])
		s_forecast = np.std(ls["price"])
		mu_change = np.mean(ls["change"])
		s_change = np.std(ls["change"])
		
		return render_template("valuation.html",name=name.upper(),invs=invs,mu_coe=mu_coe,s_coe=s_coe,mu_cod=mu_cod,s_cod=s_cod,mu_wacc=mu_wacc, s_wacc=s_wacc, mu_forecast=mu_forecast, s_forecast=s_forecast,mu_change=mu_change,s_change=s_change,price=price)
	return render_template("val-stats.html")

@app.route("/blog")
def blog_view():
    blogs = Blog.query.all()
    return render_template("blog-view.html", blogs=blogs)


@app.route("/blog/<thread>")
def blog_thread(thread):
    blogs = Blog.query.filter_by(thread=thread.lower()).all()
    return render_template("blog-view.html", blogs=blogs)

@app.route("/write/blog")
@login_required
def write_blog():
	return render_template("blog.html")

@app.route("/delete/blog/<int:id>")
def delete_blog(id):
	b = Blog.query.filter_by(id=id).first()
	db.session.delete(b)
	db.session.commit()
	return render_template("blog.html")

@app.route("/write/blog",methods=['POST'])
def submit_blog():
	title = request.values.get('title')
	content = request.values.get('content')
	blog_thread = request.values.get('thread').lower()
	file = request.files['file']
	data = file.read()
	blog = Blog(title=title,content=content,f=data,thread=blog_thread)
	db.session.add(blog)
	db.session.commit()
	return render_template("blog.html")

@app.route('/track/investment', methods=['GET','POST'])
def track_inv():
	if request.method=="POST":
		name = request.values.get('ticker').upper()
		session = requests.Session()
		session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
		# data = yf.Ticker("AAPL", session=session)
		t = yf.Ticker(name.upper(),session=session)
		price = t.history()["Close"][-1]
		invs = InvestmentDatabase.query.filter_by(investment_name=name).all()
		ls = {'spread':[],"reversion":[],"risk_neutral":[],'timedelta':[],'target_price':[]}
		for i in invs:
			ls['spread'].append(i.spread)
			ls['reversion'].append(i.reversion)
			ls['risk_neutral'].append(i.risk_neutral)
			ls['timedelta'].append(i.time_float)
			ls['target_price'].append(i.target_price)
		mu_spread = np.mean(ls["spread"])
		mu_reversion = np.mean(ls["reversion"])
		mu_risk_neutral = np.mean(ls["risk_neutral"])
		s_spread = np.std(ls["spread"])
		s_reversion = np.std(ls["reversion"])
		s_risk_neutral = np.std(ls["risk_neutral"])
		average_stoch_price = stoch_price(.01, np.mean(ls['timedelta']), mu_risk_neutral, mu_spread, mu_reversion, price, np.mean(ls['timedelta']))
		return render_template("asset.html",invs=invs,name=name,mu_spread=mu_spread,mu_reversion=mu_reversion, mu_risk_neutral=mu_risk_neutral,s_spread=s_spread,s_reversion=s_reversion,s_risk_neutral=s_risk_neutral,price=average_stoch_price)
	return render_template("track-investments.html")
	
@app.route('/submit/valuation', methods=['GET','POST'])
@login_required
def submit_valuation():
	user = current_user
	if request.method == 'POST':
		company = request.values.get('ticker').upper()
		forecast = float(request.values.get('forecast'))
		wacc = float(request.values.get('wacc'))
		roe = float(request.values.get('roe'))
		rd = float(request.values.get('rd'))
		change = float(request.values.get("change"))
		price = 0 # float(request.values.get("price"))
		file = request.files['file']
		name = request.values.get("file_name")
		file_data = file.read()
		valuation = ValuationDatabase(
			owner=user.personal_token,
			target_company=company,
			forecast = forecast,
			wacc=wacc,
			price=price,
			roe=roe,
			rd=rd,
			change_value=change,
			receipt=os.urandom(10).hex(),
			valuation_model=file_data)
		db.session.add(valuation)
		db.session.commit()
		return """<h1><a href='/'>Back</a></h1><h2>Sucessfully submit valuation</h2>"""
	return render_template("submit-val.html")

@app.route('/track/valuation',methods=['GET','POST'])
@login_required
def track_valuation():
	if request.method =="POST":
		user = current_user
		wal = WalletDB.query.filter_by(address=user.username).first()
		receipt = request.values.get("receipt")
		val = ValuationDatabase.query.filter_by(receipt=receipt).first()
		wal2 = Users.query.filter_by(username=val.owner).first()
		if val.price <= wal.coins:
			# wal.coins -= val.price
			# wal2.wallet.coins += val.price
			# db.session.commit()
			name = val.target_company
			data = val.valuation_model
			f = open('local/{name}','wb')
			f.write(data)
			f.flush()
			return send_file('local/{name}', mimetype='text/xlsx',download_name='valuation.xlsx',as_attachment=True)
		else:
			return "<h1><a href='/'>Home</a></h1><h2>Insufficient Coins in WALLET</h2>"
	return render_template("track-valuation.html")

import io
@app.route("/view/valuation",methods=["GET", "POST"])
def view_valuation():
	if request.method == "POST":
		receipt_address = request.values.get('receipt_address')
		vals = ValuationDatabase.query.filter_by(receipt=receipt_address).first()
		binary = vals.valuation_model
		xlsx = io.BytesIO(binary)
		df = pd.read_excel(xlsx)
		df = df.fillna("")
		table_html = df.to_html(classes="styled-table", index=False, escape=False)
		return render_template("view_excel.html",table_html=table_html,company=vals.target_company,val=vals)
	return render_template("valuation_view.html")

@app.route("/ledger/valuation")
def valuation_ledger():
	vals = ValuationDatabase.query.all()
	return render_template("valuation-ledger.html",invs=vals)

@app.route("/validate/valuation",methods=["GET","POST"])
def validate_val():
	if request.method == "POST":
		return 0
	return render_template("validate_valuation.html")


@app.route('/submit/optimization', methods=['GET','POST'])
@login_required
def submit_optimization():
	user = current_user
	if request.method == 'POST':
		receipt = os.urandom(10).hex()
		file = request.files['file']
		input_data = request.files['input']
		name = request.values.get("file_name")
		description = request.values.get("description")
		price = request.values.get("price")
		file_data = file.read()
		input_data = input_data.read()
		pending = PendingTransactionDatabase(
			txid=os.urandom(10).hex(),
			username = user.username,
			from_address = user.personal_token,
			to_address = "Valuation Chain",
			amount = price,
			timestamp = dt.date.today(),
			type = 'send',
			signature = receipt
		)
		db.session.add(pending)
		optimization = Optimization(
							   price=price,
							   input_data=input_data,
							   filename=name,
							   description=description,
							   created_at = dt.datetime.now(),
							   file_data=file_data,
							   receipt=receipt)
		db.session.add(optimization)
		db.session.commit()
		return "<h1><a href='/'> Back </a></h1> <h3>File successfully uploaded and saved as binary in the database.</h3>" 
	return render_template("submit-optimization.html")

@app.route("/track/opts",methods=['GET','POST'])
@login_required
def get_opts():
	user = current_user
	username = user.username
	wal = WalletDB.query.filter_by(address=username).first()
	if request.method == "POST":
		receipt = request.values.get('receipt')
		opt = Optimization.query.filter_by(receipt=receipt).first()
		if opt.price <= wal.coins:
			wal.coins-=opt.price
			db.session.commit()
			name = opt.filename
			data = opt.file_data
			f = open('local/{name}','wb')
			f.write(data)
			f.flush()
			return send_file('local/{name}', mimetype='text/py', download_name='optmization.py',as_attachment=True)
	return render_template("track-opt.html")

@app.route("/mine/optimization", methods=['GET', 'POST'])
@login_required
def mine_optimization():
	user = current_user
	wal = WalletDB.query.filter_by(address=user.username).first()
	
	if request.method == "POST":
		receipt = request.values.get("receipt")
		description = (request.values.get("description"))
		grade = int((request.values.get("grade")))
		input_data = request.files['input'].read()

		# Ensure the receipt exists in the query
		optmimization = Optimization.query.filter_by(receipt=receipt).first()
		if not optmimization:
			return """<h2>Receipt not found</h2>""", 400
		
		f = request.files['file']
		output_data = f.read()
		modified_data = request.files['file_two'].read()
		additional_data = request.files['file_three'].read()
		token = OptimizationToken(
							file_data=optmimization.file_data,
							input_data=input_data,
							receipt=receipt,
							grade=grade,
							additional_data=additional_data,
							modified_data = modified_data,
							output_data=output_data,
							string_data=output_data.decode('utf-8'),
							filename=optmimization.filename,
							created_at=dt.datetime.now(),
							description=description)
		db.session.add(token)
		db.session.commit()
		packet = {
			'index': len(blockchain.chain) + 1,
			'previous_hash': sha256(str(blockchain.get_latest_block()).encode()).hexdigest(),
			'datetime': str(dt.datetime.now()),
			'transaction': optmimization.receipt,
		}
		encoded_packet = str(packet).encode().hex()
		blockchain.add_block(packet)
		return """<h1><a href='/'>Home</a></h1><h2>Success</h2>"""
	return render_template("mine-optimization.html")

@app.route('/ledger/optimizations-tokens')
def optimizatoin_token_ledger():
	opts = OptimizationToken.query.all()
	return render_template("opt-token-ledger.html",invs=opts)

@app.route('/ledger/optimizations')
def opt_ledger():
	opts = Optimization.query.all()
	return render_template("opt-ledger.html",invs=opts)


@app.route("/generate/dcf-xlsx-template")
def generate_dcf_csv():
	return send_file('local/valuation_template.xlsx', mimetype='text/xlsx', download_name='dcf.xlsx',as_attachment=True)

@app.route('/download/csv',methods=['GET','POST'])
def download_csv():
    if request.method == "POST":
        tickers = request.values.get("tickers").upper()
        tickers = tickers.replace(',', ' ')
        tickers = yf.Tickers(tickers)
        period = request.values.get("period")
        interval = request.values.get("interval")
        history = tickers.history(period=period,interval=interval)
        df = history["Close"]
        csv = df.to_csv('data.csv')
        return send_file('data.csv', mimetype='text/csv', download_name='data.csv',as_attachment=True)
    return render_template("download-csv.html") 

@app.route("/download/return_on_capital")
def download_file():
    try:
        return send_from_directory(DOWNLOAD_FOLDER, 'return_on_capital.xlsx', as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
	with app.app_context():
		db.create_all()
		PendingTransactionDatabase.genisis() 
		app.run(host="0.0.0.0",port=2000)