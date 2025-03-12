from flask import Flask, render_template, request, redirect, abort, jsonify,sessions, Response, url_for,send_file,render_template_string,flash
from flask import Blueprint
from flask_caching import Cache
import asyncio
import socket
import os
import statsmodels.api as sm
from quart import Quart
from web3 import Web3
import os
import csv
import random
import subprocess as sp
import xml.etree.ElementTree as ET
from flask_executor import Executor
import xml.dom.minidom
import yfinance
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
from models import Swap,SwapBlock
from swap_model import TokenizedInterestRateSwap
import uuid
import logging
import schedule # apscheduler.schedulers.background import BackgroundScheduler
import threading
from twilio.rest import Client
from flask import session
from arch import arch_model
from celery import Celery
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['CACHE_TYPE'] = 'simple'  # Simple in-memory cache
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout (in seconds)
cache = Cache(app)
executor = Executor(app)
Session = scoped_session(sessionmaker(bind=engine))


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

app.config['CELERY_BROKER_URL'] = 'rediss://red-cv8uqftumphs738vdlb0:cfUOo7EcybRJpEkjPt5Fa0RkqpZA3lSg@oregon-keyvalue.render.com:6379'
app.config['CELERY_RESULT_BACKEND'] = 'rediss://red-cv8uqftumphs738vdlb0:cfUOo7EcybRJpEkjPt5Fa0RkqpZA3lSg@oregon-keyvalue.render.com:6379'

# Ensure Celery accepts SSL options
celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(
    broker_use_ssl={
        'ssl_cert_reqs': ssl.CERT_REQUIRED  # Use ssl.CERT_REQUIRED for production
    },
    result_backend_use_ssl={
        'ssl_cert_reqs': ssl.CERT_REQUIRED
    }
)

@app.teardown_appcontext
def shutdown_session(exception=None):
    Session.remove()

@app.route('/my_portfolio/<name>', methods=['GET'])
@login_required
def portfolio(name):
	user = current_user
	portfolio = Portfolio.query.filter_by(username=user.username,name=name).all()
	return render_template('portfolio.html', portfolio=portfolio)

def execute_swap():
	"""Executes swap transactions at scheduled intervals"""
	swaps = Swap.query.all()  # Get all swaps from the database

	for s in swaps:
		periods = s.amount
		maturity = s.maturity
		time_total = maturity * 365  # Convert years to days

		# Fetch historical stock data
		ticker = yf.Ticker(s.equity.upper())
		historical_data = ticker.history(period='ytd', interval='1d')["Close"]

		if len(historical_data) < 2:
			print(f"Not enough historical data for {s.equity}")
			continue  # Skip if data is insufficient

		ret = np.log(historical_data[-1]) - np.log(historical_data[-2])  # Log return
		print(f"Return for {s.equity}: {ret}")

		wallet_one = WalletDB.query.filter_by(address=s.counterparty_a).first()
		wallet_two = WalletDB.query.filter_by(address=s.counterparty_b).first()

		if not wallet_one or not wallet_two:
			print(f"Wallets not found for {s.counterparty_a} or {s.counterparty_b}")
			continue  # Skip if wallets are missing

		def logic():
			if s.status == 'Approved' and s.total_amount >= 0:
				wallet_one.swap_debt_balance += s.notional * s.fixed_rate
				wallet_two.swap_credit_balance += s.notional * abs(ret)
				s.total_amount -= (s.notional * maturity)/periods
				db.session.commit()  # Commit transaction
				print(f"Executed swap for {s.id}: Debt Balance Updated")

		# Schedule swap execution at defined intervals
		execution_interval = time_total / periods  # Calculate days per execution
		schedule.every(int(execution_interval)).days.do(logic)

	print("All swaps scheduled successfully.")
# Run the function every 10 seconds (for testing purposes)
schedule.every(100).seconds.do(execute_swap)


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
			message=message
		)
		db.session.add(notification)
		db.session.commit()
		return jsonify({"message": "Notification sent successfully"}), 201
	return render_template("send_note.html")

@app.route('/get/notes', methods=['GET','POST'])
@login_required
def get_notifications():
	if request.method == 'POST':
		u = request.form['user']
		notifications = Notification.query.filter_by(receiver_id=u).order_by(Notification.timestamp.desc()).all()
		return render_template("my_notes.html",notes=notifications)
	return render_template("get_notes.html")


@app.route('/duo-factor/requests', methods=['GET'])
@login_required
def dual_factor():
	dual = DualFactor.query.filter_by(username=current_user.username).all()
	return render_template('duo-factor.html',dual=dual)


@app.route('/swaps/request', methods=['POST','GET'])
def request_swap():
	if request.method == 'POST':
		requesting_party = request.form['requesting_party']
		counterparty_b = request.form['counterparty_b']
		counterparty_a = request.form['counterparty_a']
		floating_rate_spread=float(request.form['floating_rate_spread'])
		fixed_rate = float(request.form['fixed_rate'])
		notional = float(request.form['notional'])
		floating_rate_spread = float(request.form['floating_rate_spread'])
		issuer = counterparty_a
		periods = float(request.form['tokens'])
		receipt = os.urandom(10).hex()
		equity = request.form['equity_rate'].upper()
		id = len(Swap.query.all()) + 1 #r#andom.randint(0,1_000_000_000_000)
		sdb = Swap(id=id,notional=notional,status='Pending',fixed_rate=fixed_rate,equity=equity,amount=periods,receipt=receipt,
			 floating_rate_spread=floating_rate_spread,
			 counterparty_a=counterparty_a,counterparty_b=counterparty_b)
		# trans =  TransactionDatabase(txid=os.urandom(10).hex(),username=requesting_party,
							#    from_address=requesting_party,to_address=counterparty_b,amount=0,
							#    timestamp=datetime.utcnow,type='swap',signature=os.urandom(10).hex())
		# swp_trans = SwapTransaction(swap_id=id,receipt=os.urandom(10).hex(),
							#   sender=counterparty_a,receiver=counterparty_b,amount=periods,timestamp=dt.datetime.utcnow())
		
		duo_factor_one = DualFactor(identifier=receipt,dual_factor_signature=os.urandom(10).hex(),username=counterparty_a,
							  from_address=requesting_party,to_address=counterparty_b,amount=periods,timestamp=datetime.utcnow())
		
		duo_factor_two = DualFactor(identifier=receipt,dual_factor_signature=os.urandom(10).hex(),username=counterparty_b,
							  from_address=requesting_party,to_address=counterparty_b,amount=periods,timestamp=datetime.utcnow())
		db.session.add(duo_factor_one)
		db.session.add(duo_factor_two)
		# db.session.add(swp_trans)
		# db.session.add(trans)		
		db.session.add(sdb)
		db.session.commit()
	return render_template('request_swap.html')


                           
@app.route('/all/swaps', methods=['POST', 'GET'])
def pending_swap():
	s = Swap.query.all()
	return render_template('pending_swaps.html', swaps=s)


# def request_negotiation():
@app.route('/swaps/negotiate', methods=['POST', 'GET'])
def negotiate_swap():
	if request.method == 'GET':
		session['negotiation_signature'] = str(random.randint(0, 1_000_000))
		print(session['negotiation_signature'])

		# Secure credentials
		account_sid = 'ACbdc55c104aec01f8aae7df05d05966fb'
		auth_token = '467316527e47db772e74d8e716fcf16c'
		
		if not account_sid or not auth_token:
			return "Twilio credentials are missing", 500

		try:
			client = Client(account_sid, auth_token)
			message = client.messages.create(
				from_='+16205228999',
				body=session['negotiation_signature'],
				to='+5511994441328'
			)
		except Exception as e:
			return f"Failed to send SMS: {str(e)}", 500

		return render_template('negotiate_swap.html')

	elif request.method == 'POST':
		swap_id = request.form.get('swap_id')
		party = request.form.get('party')
		sig = request.form.get('signature')

		stored_sig = session.get('negotiation_signature')
		if stored_sig != sig:
			return "Invalid signature", 400

		try:
			notional = float(request.form['notional'])
			fixed_rate = float(request.form['fixed_rate'])
			floating_rate_spread = float(request.form['floating_rate_spread'])
		except ValueError:
			return "Invalid numerical values", 400

		swap = Swap.query.filter_by(id=swap_id).first()
		if not swap:
			return "Swap not found", 404

		# Update swap terms
		swap.notional = notional
		swap.fixed_rate = fixed_rate
		swap.floating_rate_spread = floating_rate_spread
		db.session.commit()

		session.pop('negotiation_signature', None)
		return redirect('/')

	return render_template('negotiate_swap.html')




@app.route('/swaps/approve', methods=['POST','GET'])
def approve_swap():
	if request.method == 'POST':
		swap_id = request.form['swap_id']
		approving_party = request.form['approving_party']
		second_party = request.form['destination']
		# approving_password = request.form['approving_password']
		iden = Swap.query.filter_by(id=swap_id).first()
		duo_one = DualFactor.query.filter_by(username=approving_party,identifier=iden.receipt).first()#.all()[-1]
		duo_two = DualFactor.query.filter_by(username=second_party,identifier=iden.receipt).first()#all()[-1]
		dual_one = request.form['dual_one']
		dual_two = request.form['dual_two']
		
		print('1\t',duo_one.dual_factor_signature,'\n2\t',dual_one,'\n3\t',dual_two,'\n4\t',duo_two.dual_factor_signature)

		if duo_one.dual_factor_signature == dual_one and duo_two.dual_factor_signature == dual_two :
			s = Swap.query.filter_by(id=swap_id).first()
			s.status = 'Approved'
			db.session.commit()
			db.session.delete(duo_one)
			db.session.delete(duo_two)
			db.session.commit()
			transaction = SwapTransaction(id=s.id,swap_id=swap_id,receipt=s.receipt,
									sender=s.counterparty_a,receiver=s.counterparty_b,
									amount=s.amount,status='approved',timestamp=datetime.now())
			db.session.add(transaction)
			db.session.commit()
			wallet_one = WalletDB.query.filter_by(address=approving_party).first()
			wallet_two = WalletDB.query.filter_by(address=second_party).first()
			wallet_one.swap_debt_balance -= s.notional
			db.session.commit()
			return redirect('/swap/market')
	return render_template('approve_swap.html')


@app.route('/swaps/reject', methods=['POST','GET'])
def reject_swap():
	if request.method == 'POST':
		swap_id = request.form['swap_id']
		rejecting_party = request.form['rejecting_party']
		s = Swap.query.filter_by(id=swap_id).first()
		db.session.delete(s)
		db.session.commit()
		return redirect('/swap/market')
	return render_template('reject_swap.html')


@app.route('/swap/market')
def swap_marketplace():
	swaps = Swap.query.all()
	return render_template('swap_manage.html',swaps=swaps)

@app.route('/swap/index')
def index_swap():
	swaps = Swap.query.all()
	return render_template('swap_index.html', swaps=swaps)


@app.route('/stoch/greeks', methods=['GET','POST'])
@cache.cached(300)  # Cache this route for 5 minutes
def calc_greeks():
	ls = []
	df = {"name":[],"delta": [], "gamma": [], "theta": [], "vega": [], "rho": [], "price": [],'receipt': []}
	invests = InvestmentDatabase.query.all()
	for i in invests:
		t = yf.Ticker(i.investment_name.upper())
		price = t.history(period='1d')['Close'].iloc[-1]
		greeks = calculate_greeks(1/52, i.time_float, i.risk_neutral, i.spread, i.reversion, price, i.target_price)
		df["name"].append(i.investment_name)
		df["delta"].append(greeks['Delta'])
		df["gamma"].append(greeks['Gamma'])
		df["theta"].append(greeks["Theta"])
		df["vega"].append(greeks["Vega"])
		df["rho"].append(greeks["Rho"])
		df["price"].append(greeks["Price"])
		df["receipt"].append(i.receipt)

	html = pd.DataFrame(df).to_html(index=False)
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

@app.route('/option/greeks', methods=['GET','POST'])
def calc_option_greeks():
	from greeks import black_scholes_greeks
	ls = []
	df = {"name":[],"delta": [], "gamma": [], "theta": [], "vega": [], "rho": [],'price':[],"receipt":[]}
	invests = InvestmentDatabase.query.all()
	for i in invests:
		try:
			t = yf.Ticker(i.investment_name.upper())
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
			t = yf.Ticker(i.investment_name.upper())
			prices_vector = t.history(period='5d',interval='1m')
			price = t.history()['Close'].iloc[-1]
			s = stoch_price(1/52, i.time_float, i.risk_neutral, i.spread, i.reversion, price, i.target_price)
			i.stoch_price = s
			# i.tokenized_price
			db.session.commit()
		except:
			pass

def change_value_update():
	invests = InvestmentDatabase.query.all()
	for i in invests:
		t = yf.Ticker(i.investment_name.upper())
		prices_vector = t.history(period='5d',interval='1m')
		price = t.history(period='1d',interval='1m')['Close'].iloc[-1]
		change = np.log(price) - np.log(i.starting_price)
		i.change_value = change
		db.session.commit()
@celery.task	
def update():
	"""
	Update investment data, including market price, time_float, tokenized price, and coins.
	"""
	print("Update")
	recalculate()
	change_value_update()
	invests = InvestmentDatabase.query.all()

	for i in invests:
		try:
			# Fetch current market data
			t = yf.Ticker(i.investment_name.upper())
			prices_vector = t.history(period='5d', interval='1m')
			price = t.history(period='1d', interval='1m')['Close'].iloc[-1]
			
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

@app.route('/long_task')
def long_task():
    executor.submit(update)
    return "Task started"

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

@app.route('/peer/chat')
def chat_html():
	return render_template("chatjs.html")


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

@app.route('/login', methods=['POST','GET'])
def login():
	if request.method == "POST":
		username = request.values.get("username")
		password = request.values.get("password")
		user = Users.query.filter_by(username=username).first()
		if user and bcrypt.check_password_hash(user.password, password):
			login_user(user)
			return redirect('/')
		else:
			return redirect('/signup')
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
										 type='SEND')
			db.session.add(new_transaction)
			db.session.commit()
			coin_db = CoinDB.query.get_or_404(1)
			# coin_db.gas(blockchain,6)
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
	return """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""


@login_required
@app.route('/html/investment/ledger',methods=['GET'])
def html_investment_ledger():
	# update()
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
def add_portfolio():
	if request.method == 'POST':
		ticker = request.form.get('name').upper()
		name = request.form.get('pname').lower()
		price = yf.Ticker(ticker).history(period='ytd',interval='1d')['Close']
		mean = np.mean(price)
		std = np.std(price)
		weight = float(request.form.get('weight'))
		user = request.form.get("username")
		u = Users.query.filter_by(username=user).first()
		prt = Portfolio(name=name,mean=mean,std=std,weight=weight,price=price[-1],username=user,
				token_name=ticker.upper(),token_address=os.urandom(10),user_address=u.personal_token,transaction_receipt=os.urandom(10))
		db.session.add(prt)
		db.session.commit()
		return f"""<a href='/'><h1>Home</h1></a><h3>Success</h3>Portfolio {ticker}"""
	return render_template("add_port.html")


@app.route('/holdings', methods=['GET'])
@login_required
def get_user_wallet():
	update.delay()
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

from flask import request, render_template
import numpy as np
import datetime as dt
import os
from hashlib import sha256
import yfinance as yf
from pricing_algo import derivative_price
from bs import black_scholes
from algo import stoch_price
from models import Users, WalletDB, TransactionDatabase, AssetToken, InvestmentDatabase, PendingTransactionDatabase, Block, db
import scipy

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

            invest_name = invest_name.upper()
            coins, qt, target_price, maturity = map(float, [coins, qt, target_price, maturity])
            risk_neutral, spread, reversion = map(float, [risk_neutral, spread, reversion])
            option_type = option_type.lower()

            user_db = Users.query.filter_by(username=user).first()
            if not user_db:
                return "<h3>User not found</h3>"

            ticker = yf.Ticker(invest_name)
            history = ticker.history(period='1d', interval='1m')

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
				if (wal.coins >= staked_coins) and (inv.quantity > staked_coins): # and (inv.coins_value >= staked_coins):
					inv.quantity -= staked_coins
					db.session.commit()
					total_value = inv.tokenized_price * staked_coins
					house = BettingHouse.query.get_or_404(1)
					house.coin_fee(0.1 * total_value)
					owner_wallet.coins += 0.1 * total_value
					db.session.commit()
					new_value = 0.8 * total_value
					wal.coins -= total_value
					inv.tokenized_price += new_value
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


@app.route('/asset/info/<int:id>')
def info_assets(id):
	update.delay()
	asset = InvestmentDatabase.query.get_or_404(id)
	return render_template("asset-info.html", asset=asset)

@app.route('/get/asset/<int:id>',methods=['GET','POST'])
def get_asset(id):
	try:
		t = InvestmentDatabase.query.get_or_404(id)
		info = {'id': t.id,'name': str(t.investment_name),'owner':t.owner,'investors_num':t.investors,'market_cap':str(t.market_cap),'coins_value':str(t.coins_value),'receipt':str(t.receipt),'tokenized_price':str(t.tokenized_price),'market_price':t.market_price,'change':t.change_value,'original_price':t.starting_price}
		return jsonify(info)
	except:
		return "<h2>The asset is no longer active<h2>"

@app.route('/price',methods=['GET','POST'])
def price():
	if request.method =="POST":
		username = request.values.get('username')
		password = request.values.get('password')
		stake = float(request.values.get("stake"))
		S = float(request.form['S'])
		K = float(request.form['K'])
		T = float(request.form['T'])
		r = float(request.form['r'])
		sigma = float(request.form['sigma'])
		option_type = request.form['option_type']
		price = black_scholes(S, K, T, r, sigma)
		return f"{price}"
	return render_template("options-pricing.html")

@app.route('/greeks',methods=['GET','POST'])
def greeks():
	from greeks import black_scholes_greeks
	if request.method =="POST":
		S = float(request.form['S'])
		K = float(request.form['K'])
		T = float(request.form['T'])
		r = float(request.form['r'])
		sigma = float(request.form['sigma'])
		option_type = request.form['option_type']
		greeks = black_scholes_greeks(S, K, T, r, sigma)
		return jsonify(greeks)
	return render_template("options-pricing.html")

@app.route('/stoch-greeks',methods=['GET','POST'])
def stochGreeks():
	from stoch_greeks import calculate_greeks
	if request.method =="POST":
		s0 = float(request.form['S'])
		dt = float(request.form['dt'])
		k = float(request.form['K'])
		t = float(request.form['T'])
		r = float(request.form['r'])
		sigma = float(request.form['sigma'])
		mu = float(request.form['mu'])
		option_type = request.form['option_type']
		greeks = calculate_greeks(dt, t, r, sigma, mu, s0, k, option_type='call')
		return jsonify(greeks)
	return render_template("stoch-greeks.html")

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

@app.route('/bib')
def bib():
	return render_template("bib-template.html")

@app.route('/render')
def render():
	return render_template("index_render.html")

# Function to check allowed file extensions
def allowed_file(filename):
	from models import ALLOWED_EXTENSIONS
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'html','py','pdf'}

import math
@app.route("/neutral-measure",methods=['GET','POST'])
def neutral_measure():
	if request.method == 'POST':
		company = request.values.get('ticker').upper()
		interval = request.values.get("interval")
		period = request.values.get("period")
		
		def risk_neutral_probability(r, u, d, delta_t=1/252):
			risk_free_growth = math.exp(r * delta_t)
			q = (risk_free_growth - d) / (u - d)
			return q

		def bin_stats_df(t,period='1d',interval='1m',rf=.05,dt=1/252):
			ticker = yf.Ticker(t.upper())
			history = ticker.history(period=period,interval=interval)
			df = history[['Close','Open']]
			up = []
			down = []
			for i in range(0, len(df)):
				direction = df["Close"][i] - df['Open'][i]
				if direction > 0:
					up.append(direction)
				else:
					down.append(direction)
					
			prob_up = len(up)/(len(up)+len(down))
			mean_up = np.mean(up)
			std_up = np.std(up)
			prob_down = len(down)/(len(up)+len(down))
			mean_down = np.mean(down)
			std_down = np.std(down)
			
			def risk_neutral_probability(r, u, d, delta_t=dt):
				risk_free_growth = math.exp(rf * delta_t)
				q = (risk_free_growth - d) / (u - d)
				return q
			
			result = risk_neutral_probability(rf, 1+mean_up, 1+mean_down, dt)
			return result #pd.DataFrame({'rnp':[result],'prob_up':[prob_up],'up_factor':[1 + mean_up], 'prob_down': [prob_down], 'down_factor': [1 + mean_down]},index=[t])
		res = bin_stats_df(company,period=period,interval=interval,)
		return render_template('neutral_measure.html',risk_neutral=res)
	return render_template('neutral_measure.html')

@app.route("/indicator-measure",methods=['GET','POST'])
def risk_neutral_measure():
	if request.method == 'POST':
		def indicator(x, threshold):
			return 1 if x > threshold else 0
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		t = yf.Tickers(tickers)
		h = t.history(start='2020-1-1',end=dt.date.today(),interval="1d")
		df = h["Close"].pct_change()[1:]
		cov = df.cov()
		covariance = np.matrix(cov)*np.sqrt(256)
		mu = np.matrix(df.mean()).T*np.sqrt(256)
		sig = np.matrix(df.std()).T*np.sqrt(256)
		results = np.zeros((4, 10000))
		for i in range(10000):
			weights = np.random.random(len(mu))
			weights /= np.sum(weights)
			portfolio_return = np.dot(weights, mu)
			portfolio_stddev = (weights * covariance)@weights.T
			sharpe_ratio = (portfolio_return) / portfolio_stddev
			results[0,i] = portfolio_return
			results[1,i] = portfolio_stddev
			results[2,i] = sharpe_ratio
			results[3,i] = i
		results_df = pd.DataFrame(results.T, columns=['Returns', 'Volatility', 'SharpeRatio', 'Portfolio Index'])
		deviation = results_df["SharpeRatio"].std()
		mean = results_df["SharpeRatio"].mean()
		status = []
		for i in range(len(results_df)):
			sharp = results_df["SharpeRatio"][i]
			r = indicator(sharp, mean)
			status.append(r)
		results_df["indicator"] = status
		results_df.to_csv("results.csv")
		i_prob = results_df["indicator"].loc[results_df["indicator"] == 1]
		ic_prob = results_df["indicator"].loc[results_df["indicator"] == 0]
		risk_netural = len(i_prob)/(len(i_prob)+len(ic_prob))
		return render_template("risk-neutral-measure.html",risk_netural=risk_netural)
	return render_template("risk-neutral-measure.html")

@app.route('/valuation/statistics', methods=['GET','POST'])
def valuation_stats():
	if request.method=="POST":
		name = request.values.get('ticker')
		t = yf.Ticker(name.upper())
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
		t = yf.Ticker(name.upper())
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
		name = request.values.get("file_name")
		description = request.values.get("description")
		price = request.values.get("price")
		file_data = file.read()
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
		db.session.commit()
		optimization = Optimization(
							   price=price,
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
		wal.coins += 50
		db.session.commit()
		receipt = request.values.get("receipt")
		description = (request.values.get("description"))
		grade = int((request.values.get("grade")))

		# Ensure the receipt exists in the query
		optmimization = Optimization.query.filter_by(receipt=receipt).first()
		if not optmimization:
			return """<h2>Receipt not found</h2>""", 400
		f = request.files['file']
		output_data = f.read()
		token = OptimizationToken(
							file_data=optmimization.file_data,
							receipt=receipt,
							grade=grade,
							output_data=output_data,
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

@app.route('/process/optimization', methods=['GET', 'POST'])
def process_optimization():
	if request.method == 'POST':
		receipt_address = request.values.get('receipt')
		return receipt_address
	return render_template('process-optimization.html')

@app.route('/ledger/optimizations')
def opt_ledger():
	opts = Optimization.query.all()
	return render_template("opt-ledger.html",invs=opts)


##################################################
# Quantitative Services #########################
##################################################

@app.route("/html-dcf",methods=["GET",'POST'])
def html_dcf():
	return render_template('dcf.html')

@app.route("/equity-risk-premium",methods=["GET",'POST'])
def IERP():
	from ierp import calculate_implied_equity_risk_premium
	if request.method=="POST":
		div = float(request.values.get('div'))
		market = float(request.values.get('market'))
		growth = float(request.values.get('g'))
		rf = float(request.values.get('rf'))
		result = calculate_implied_equity_risk_premium(div,market,growth,rf)
		return render_template('ierp.html',result=result)
	return render_template('ierp.html')

@app.route("/forward-rate-empirical",methods=["GET",'POST'])
def forward_rate_two():
	from forward_rate import f
	if request.method=="POST":
		p = request.values.get('period')
		i = request.values.get('interval')
		ticker = request.values.get("ticker").upper()
		t = yf.Ticker(ticker)
		history = t.history(period=p,interval=i)
		
		close_prices = history["Close"].pct_change()[1:]
		open_prices = history["Open"].pct_change()[1:]
		
		c = [(i-np.mean(close_prices))/np.std(close_prices) for i in close_prices]
		o = [(i-np.mean(open_prices))/np.std(open_prices) for i in open_prices]
		
		ls = []
		# Loop through open and close prices, calculating f(t, r) for each day
		for open_price, close_price in zip(c, o):
			result = f(close_price, open_price)
			ls.append(result)
		r = np.array(ls)
		forward_rate = (r@r/np.linalg.norm(r))/100
		print(forward_rate)
		print('Coefficient\t',r)
		
		return render_template('forward-rate-two.html',forward=forward_rate)
	return render_template('forward-rate-two.html')


@app.route("/transform_coef",methods=["GET",'POST'])
def transform_coef():
	from f2 import f
	if request.method=="POST":
		p = request.values.get('period')
		i = request.values.get('interval')
		ticker = request.values.get("ticker").upper()
		t = yf.Ticker(ticker)
		history = t.history(period=p,interval=i)
		
		hig_low = history["High"] - history["Low"]
		hl = hig_low.pct_change()[1:]# Percentage returns (ignoring first NaN)
		hl = [(i-np.mean(hl))/np.std(hl) for i in hl]
		
		
		volume = history["Volume"]
		v1 = volume.pct_change()[1:]# Percentage returns (ignoring first NaN)
		v1 = [(i-np.mean(v1))/np.std(v1) for i in v1]
		
		open = history["Open"]
		ret2 = open.pct_change()[1:]# Percentage returns (ignoring first NaN)
		ret2 = [(i-np.mean(ret2))/np.std(ret2) for i in ret2]
		
		spread = history["Close"] - history["Open"]
		ret3 = spread#.pct_change()[1:]# Percentage returns (ignoring first NaN)
		ret3 = [(i-np.mean(ret3))/np.std(ret3) for i in ret3]
		
		prices = history["Close"]
		ret = prices.pct_change()[1:]
		ret = [(i-np.mean(ret))/np.std(ret) for i in ret]
	
		vol = np.array([f(i) for i in hl])
		vol = np.sqrt(vol@vol)
		
		volm = np.array([f(i) for i in v1])
		volm = np.sqrt(volm@volm)
		
		r2 = np.array([f(i) for i in ret2])
		r2 = np.sqrt(r2@r2)
		
		r3 = np.array([f(i) for i in ret3])
		r3 = np.sqrt(r3@r3)
		
		r = np.array([f(i) for i in ret])
		r = np.sqrt(r@r)
		print('Coefficient\t',r)
		return render_template('transform_coef.html',r=r.item(), r3=r3.item(), r2=r2.item(),volume=volm,vol=vol)
	return render_template('transform_coef.html')

@app.route("/single-reversion-coef",methods=["GET",'POST'])
def reversion():
	if request.method=="POST":
		def calculate_reversion_coefficient_sklearn(prices):
			"""
			Calculate the reversion coefficient of a stock based on its price data using sklearn.
			
			Args:
			- prices (pd.Series or list): A series of historical stock prices.
			
			Returns:
			- float: The reversion coefficient.
			"""
			# Convert to pandas Series if it's not already
			if not isinstance(prices, pd.Series):
				prices = pd.Series(prices)
			
			# Prepare lagged and current prices
			lagged_prices = prices.shift(1).dropna().values.reshape(-1, 1)  # Lagged prices as X
			current_prices = prices[1:].values.reshape(-1, 1)               # Current prices as y

			# Fit the linear regression model
			model = LinearRegression()
			model.fit(lagged_prices, current_prices)
			
			# Reversion coefficient is the slope of the lagged price
			reversion_coefficient = model.coef_[0][0]
			
			return reversion_coefficient
		
		ticker = request.values.get("tickers")
		period = request.values.get("period")
		interval = request.values.get("interval")
		
		t = yf.Ticker(ticker)
		history = t.history(period=period,interval=interval)
		prices = history["Close"]
		reversion_coefficient = calculate_reversion_coefficient_sklearn(prices)

		return render_template('single-reversion.html',reversion=reversion_coefficient.item())
	return render_template('single-reversion.html')

@app.route('/ddm',methods=["GET","POST"])
def ddm():
	from wacc import Rates
	if request.method =='POST':
		ticker = request.values.get('ticker').upper()
		rf = float(request.values.get('rf'))
		erp = float(request.values.get('erp'))
		cs = float(request.values.get('cs'))
		interest_coverage = get_interestCoverage(ticker)
		taxes = get_taxes(ticker)
		reg = get_beta(ticker)
		rd = get_costDebt(ticker)
		debt   = get_debt(ticker)
		equity = get_equity(ticker)
		rate = Rates(debt, equity, taxes)
		ke = rate.re(reg, rf, erp, cs)
		costOfCapital = rate.wacc(rd,ke)
		current_price = get_price(ticker) 
		div_growth = implied_div_growth_rate(ke, get_dps(ticker), current_price)
		ddm = (get_dps(ticker)*(1+div_growth))/(ke - div_growth)
		string = f'''<!DOCTYPE html>
					<html lang="en">

					<head>
						<meta charset="UTF-8">
						<meta name="viewport" content="width=device-width, initial-scale=1.0">
						<title>Financial Metrics</title>
						<style>
							body {{
								font-family: 'Roboto', sans-serif;
								margin: 0;
								padding: 20px;
								background-color: #f4f4f9;
								color: #333;
								text-align: center;
							}}

							.metric-container {{
								display: flex;
								flex-direction: column;
								align-items: center;
								gap: 20px;
							}}

							h1 {{
								font-size: 2.5rem;
								font-weight: bold;
								margin: 0;
								padding: 10px 20px;
								background: #007bff;
								color: #fff;
								border-radius: 8px;
								box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
							}}

							.metric-title {{
								font-size: 1.5rem;
								color: #007bff;
								margin-top: 20px;
								font-weight: 600;
							}}
						</style>
					</head>

					<body>
					<h1> {ticker}</h1>
						<div class="metric-container">
							<div>
								<span class="metric-title">Discounted Dividend Model (DDM):</span>
								<h1 id="ddm">{ddm}</h1>
							</div>
							<div>
								<span class="metric-title">Dividend Growth Rate:</span>
								<h1 id="div-growth">{div_growth}</h1>
							</div>
						</div>
					</body>

					</html>
					'''
		return render_template_string(string)#f"<h1>{ddm}</h1><h1>{div_growth}</h1>"
	return render_template("ddm.html")

@app.route('/api/stocks/<symbol>', methods=['GET','POST'])
def get_stock(symbol):
	try:
		stock = yf.Ticker(symbol)
		stock_info = stock.info
		print(stock_info)
		
		# Extract relevant stock data
		stock_data = {
			'symbol': stock_info.get('symbol', symbol),
			'price': stock_info.get('currentPrice', 'N/A'),
			'change': f"{stock_info.get('regularMarketChangePercent', 'N/A'):.2f}%"
		}
		
		return jsonify(stock_data)
	except Exception as e:
		# In case of any error, return a 500 status code with error message
		return jsonify({'error': str(e)}), 500

@app.route("/basic/dcf",methods=['POST','GET'])
def basic_dcf():
	from dcf3 import DCF
	from wacc import Rates  
	from ProfMain import ProfiMain
	
	if request.method == "POST":
		ticker = request.values.get("ticker").upper()

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
		
		g = ProfiMain()
		growth = np.mean(g.roc(ticker))
		reg_beta = float(request.values.get('rg'))
		taxes = float(request.values.get('taxes'))
		rf = float(request.values.get('rf'))
		erp = float(request.values.get('erp'))
		cs = float(request.values.get('cs'))
		rd = float(request.values.get('rd'))
		dep = get_dep(ticker)/get_rev(ticker)
		debt = get_debt(ticker)
		equity = get_equity(ticker)
		cash = get_cash(ticker)
		shares = get_shares_two(ticker)
		d = Rates(debt, equity, taxes)
		re = d.re(reg_beta, rf, erp, cs)
		wacc = d.wacc(rd, re)
		netInvestment = get_ni(ticker)
		dcf= DCF(get_ni(ticker), dep, get_rev(ticker), growth, wacc, taxes)
		rev = dcf.rev()
		df = pd.DataFrame(dcf.sheet(rev),index=['operating income','net investment','D&A','taxes'],columns=[1,2,3,4,5]).T
		fcff = dcf.calculate_cashflow(rev)
		df['fcff'] = fcff
		df = df.T
		html = df.to_html()
		final = dcf.final(dcf.calculate_cashflow(rev))
		fcff = (final - get_debt(ticker) + get_cash(ticker))/get_shares_two(ticker)
		tabel =f"""<h1>{ticker}</h1>{html}<br><h3>Share Price </h3><h4>{fcff}</h4>"""
		
		return render_template_string(tabel)
	return render_template("basic-dcf.html")

@app.route('/coef-calibration',methods=["GET","POST"])
def calibration():
	import math
	def eucdist(Xk,Yk):
		return np.sqrt(sum([(Xk[i] - Yk[i])**2 for i in range(len(Xk))]))*(np.linalg.norm(Xk)*np.linalg.norm(Yk))
	if request.method == "POST":
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		t = yf.Tickers(tickers)
		h = t.history(start='2020-1-1',end=dt.date.today(),interval="1d")
		df = h["Close"].pct_change()[1:]
		cov = df.cov()
		covariance = np.matrix(cov)*np.sqrt(256)
		mu = np.matrix(df.mean()).T*np.sqrt(256)
		sig = np.matrix(df.std()).T*np.sqrt(256)
		spread = eucdist(mu,sig)
		def epsilon_ball(Xk,Yk,omega,lmd,dt=0.01):
			interval = max(Xk) - min(Yk)
			start = - abs(interval - lmd*eucdist(Xk, Yk)**omega)
			end = interval + lmd*eucdist(Xk, Yk)**omega,
			space = np.arange(start=start, stop=end[0], step=dt)
			return space*np.linalg.norm(Xk)*np.linalg.norm(Yk)
		e_ball = epsilon_ball(np.log(abs(mu)), np.log(abs(sig)), 3, .3, dt=.1)
		def poisson(lmd,x):
			return (lmd**x)*np.exp(-lmd)/math.factorial(x)
		Ak = np.matrix(e_ball*poisson(np.mean(mu),2))#*expected_indicator
		reversion = (1/np.linalg.norm(Ak))*1000000
		return render_template('calibration.html',spread=spread.item(),reversion=reversion)
	return render_template('calibration.html')
		
@app.route("/single-stock-risk-neutral", methods=["GET","POST"])
def single_risk_neutral():
	import math
	def risk_neutral_probability(r, u, d, delta_t=1):
		risk_free_growth = math.exp(r * delta_t)
		q = (risk_free_growth - d) / (u - d)
		return q
	if request.method == "POST":
		r = float(request.values.get("r"))
		u = float(request.values.get("u"))
		d = float(request.values.get("d"))
		delta = float(request.values.get("delta"))
		risk_neutral = risk_neutral_probability(r,u,d,delta)
		print(risk_neutral)
		return render_template("stock-risk-neutral.html",risk_neutral=risk_neutral)
	return render_template("stock-risk-neutral.html")
			
@app.route('/mu-sigma', methods=["GET","POST"])
def mu_sigma():
	if request.method=="POST":
		ticker = request.values.get("ticker").upper()
		period = request.values.get("period")
		interval = request.values.get("interval")
		t = yf.Ticker(ticker)
		h = t.history(period=period,interval=interval)
		df = h["Close"] 
		ret = df.pct_change()[1:]
		mu = ret.rolling(3).mean()
		sig = ret.rolling(3).std()
		X = np.vstack((mu[3:],sig[3:])).T
		y = np.matrix(ret[2:-1]).T
		lr  = LinearRegression(fit_intercept=True)
		fit = lr.fit(X, np.asarray(y))
		score = fit.score(X, np.asarray(y))
		x = fit.coef_
		exp_mu = x[0][0]
		exp_sig = x[0][1]
		pred = x@np.matrix([mu[-1],sig[-1]]).T
		price = df[-1]*(1+pred.item())
		return render_template("mu-sigma.html",rate=pred.item()*100,score=score,price=price,mu=exp_mu.item(), sigma=exp_sig.item())
	return render_template("mu-sigma.html")		
			
@app.route('/single-stock-calibration',methods=["GET","POST"])
def single_calibration():
	import math
	def eucdist(Xk,Yk):
		return np.sqrt(sum([(Xk[i] - Yk[i])**2 for i in range(len(Xk))]))*(np.linalg.norm(Xk)*np.linalg.norm(Yk))
	if request.method == "POST":
		ticker = request.values.get("tickers").upper()
		def eucdist(Xk,Yk):
			return np.sqrt(sum([(Xk[i] - Yk[i])**2 for i in range(len(Xk))]))*(np.linalg.norm(Xk)*np.linalg.norm(Yk))
		t = yf.Ticker(ticker)
		h = t.history(period='max',interval="1d")
		o = h["Open"].pct_change()[1:]
		c = h["Close"].pct_change()[1:]
		spread = eucdist(c, o)
		o = h["Open"].pct_change()[1:].rolling(7).std()[7:]
		c = h["Close"].pct_change()[1:].rolling(7).std()[7:]
		spread2 = eucdist(c, o)
		o = h["Open"].pct_change()[1:].rolling(5).mean()[5:]
		c = h["Close"].pct_change()[1:].rolling(5).mean()[5:]
		spread3 = eucdist(c, o)
		return render_template('singel-calibration.html',spread=spread.item(), spread2=spread2.item(), spread3=spread3.item())
	return render_template('singel-calibration.html')

@app.route('/live/dcf')
def live_dcf():
	return render_template('live-dcf.html')

@app.route('/live/fcfe')
def live_fcfe():
	return render_template('live-fcfe.html')

@app.route('/live/valuations')
def live():
	return render_template("live-value.html")

@app.route("/generate/dcf-xlsx-template")
def generate_dcf_csv():
	return send_file('local/valuation-template.xlsx', mimetype='text/xlsx', download_name='dcf.xlsx',as_attachment=True)

@app.route('/fundamentals', methods=['GET','POST'])
def get_fundamentals():
	if request.method == "POST":
		# Get the ticker from the query parameters
		ticker = request.values.get('ticker').upper()
		if not ticker:
			return jsonify({"error": "Ticker parameter is required"}), 400
		
		try:
			# Fetch the depreciation and revenue
			interest_coverage = get_interestCoverage(ticker)
			depreciation = get_dep(ticker)/get_rev(ticker)
			revenue = get_rev(ticker)
			shares = get_shares_two(ticker)
			debt = get_debt(ticker)
			equity = get_equity(ticker)
			cash = get_cash(ticker)
			ni = get_ni(ticker)
			gpr = get_grossProfitRatio(ticker)
			taxes = get_taxes(ticker)
			operating_income = get_operating_income(ticker)
			capex = get_capex(ticker)
			beta = get_beta(ticker)
			cost_debt = get_costDebt(ticker)
			# Return the results as JSON
			results = {
				"ticker": ticker,
				"depreciationAndAmortization(%)": depreciation,
				"revenue": revenue,
				"shares":shares,
				"debt":debt,
				"equity":equity,
				"cash":cash,
				"netIncome(%)":ni,
				"grossProfitRatio(%)":gpr,
				"taxes(%)":taxes,
		        "operatingIncome":operating_income,
				'interest_coverage':interest_coverage,
				"capex":capex,
				"beta":beta,
				"cost_debt":cost_debt}
			return render_template("fundamentals.html",results=results)
		except Exception as e:
			return jsonify({"error": str(e)}), 500
	return render_template("fundamentals.html")

@app.route('/wacc',methods=['GET','POST'])
def wacc():
	from wacc import Rates
	if request.method =="POST":
		debt = float(request.values.get('debt'))
		equity = float(request.values.get('equity'))
		taxes = float(request.values.get('taxes'))
		rg = float(request.values.get('rg'))
		rf = float(request.values.get('rf'))
		erp = float(request.values.get('erp'))
		cs = float(request.values.get('cs'))
		r = Rates(debt,equity,taxes)
		ke = r.re(rg,rf,erp,cs)
		kd = rf+cs
		wacc = r.wacc(kd,ke)
		return render_template('wac-output.html',ke=ke,kd=kd,wacc=wacc)
	return render_template('wacc.html')

@app.route("/implied-vol",methods=["POST",'GET'])
def implied_vol():
	from vol import implied_volatility_option
	if request.method == "POST":
		market = float(request.values.get("market"))
		S = float(request.values.get("S"))
		K = float(request.values.get("K"))
		T = float(request.values.get("T"))
		r = float(request.values.get("r"))
		otype = request.form['option_type']
		iv = implied_volatility_option(S,K,T,r,market,otype)
		return f"""<h1><a href='/'>Back</a></h1><h2> IMPLIED VOL</h2><h3>{iv}</h3>"""
	return render_template("IV.html")

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


@app.route('/cov/prices',methods=['GET','POST'])
def cov_prices():
	if request.method == 'POST':
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		tickers = yf.Tickers(tickers)
		history = tickers.history(start='2018-1-1',end=dt.date.today())
		df = history['Close']
		covaraince = df.cov()
		html = covaraince.to_html()
		return f"""<h1><a href='/'>Back</a></h1><h2> Covaraince Prices</h2>{html}"""
	return render_template("cov.html")

@app.route('/cov/returns',methods=['GET','POST'])
def cov_returns():
	if request.method == 'POST':
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		tickers = yf.Tickers(tickers)
		history = tickers.history(start='2018-1-1',end=dt.date.today())
		df = history['Close'].pct_change()
		covaraince = df.cov()*np.sqrt(256)
		html = covaraince.to_html()
		return f"""<h1><a href='/'>Back</a></h1><h2>Annualized Covaraince Returns</h2>{html}"""
	return render_template("cov.html")

@app.route('/corr/returns',methods=['GET','POST'])
def corr_returns():
	if request.method == 'POST':
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		tickers = yf.Tickers(tickers)
		history = tickers.history(start='2018-1-1',end=dt.date.today())
		df = history['Close'].pct_change()
		correlation = df.corr()#*np.sqrt(256)
		html = correlation.to_html()
		return f"""<h1><a href='/'>Back</a></h1><h2>Correlation Returns</h2>{html}"""
	return render_template("cov.html")

@app.route('/corr/prices',methods=['GET','POST'])
def corr_prices():
	if request.method == 'POST':
		tickers = request.values.get("tickers").upper()
		tickers = tickers.replace(',', ' ')
		tickers = yf.Tickers(tickers)
		history = tickers.history(start='2018-1-1',end=dt.date.today())
		df = history['Close']#.pct_change()
		correlation = df.corr()#*np.sqrt(256)
		html = correlation.to_html()
		return f"""<h1><a href='/'>Back</a></h1><h2>Correlation Prices</h2>{html}"""
	return render_template("cov.html")

@app.route('/graph', methods=['GET', 'POST'])
def graph():
	if request.method == 'POST':
		ticker = request.values.get("asset").upper()
		period = request.form.get("ttype")
		ticker_data = yf.Ticker(ticker)
		hist = ticker_data.history(period=period)['Close']
		# Create the Bokeh plot
		p = figure(title=f"{ticker} Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
		p.line(hist.index, hist.values, line_width=2)
		# Generate the HTML for the plot
		html = file_html(p, CDN, f"{ticker} Closing Prices")
		return html
	return render_template('graph.html')

@app.route('/graph/freq', methods=['GET', 'POST'])
def graph_day():
	if request.method == 'POST':
		ticker = request.values.get("asset").upper()
		period = request.form.get("ttype")
		ticker_data = yf.Ticker(ticker)
		hist = ticker_data.history(interval=period,period='1d')['Close']
		open = ticker_data.history(interval=period,period='1d')['Open']
		sma = hist.rolling(7).mean()
		# Create the Bokeh plot
		p = figure(title=f"{ticker} Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
		p.line(hist.index, hist.values, line_width=2)
		p.line(open.index, open.values, line_width=2,color='red')
		p.line(open.index, sma.values, line_width=2,color='green')
		# Generate the HTML for the plot
		html = file_html(p, CDN, f"{ticker} Closing Prices")
		return html
	return render_template('graph-high.html')

@app.route('/1m/forecast', methods=['GET', 'POST'])
def graph_forecast_1m():
	if request.method == 'POST':
		ticker = request.values.get("asset").upper()
		ticker_data = yf.Ticker(ticker)
#		ttype = request.form.get('')
		hist = ticker_data.history(interval='1d',period='1mo')['Close']
		initial_price = hist[-1]
		ret = hist.pct_change()[1:]
		drift = np.mean(ret)*np.sqrt(256)
		volatility = np.std(hist)#*np.sqrt(256)
		dt = 1/31
		T = 2
		price_paths = []
		for i in range(0, 100):
			price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)
		# Create the Bokeh plot
		p = figure(title=f"{ticker} Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[0], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[1],line_width=2,color='red')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[2],line_width=2,color='green')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[3], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[4],line_width=2,color='blue')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[5],line_width=2,color ='yellow')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[6], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[7],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[8],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[9], line_width=2)
		html = file_html(p, CDN, f"{ticker} Closing Prices")
		return html
	return render_template('graph-forecast.html')

@app.route('/1d/forecast', methods=['GET', 'POST'])
def graph_forecast_1d():
	if request.method == 'POST':
		ticker = request.values.get("asset").upper()
		ticker_data = yf.Ticker(ticker)
		hist = ticker_data.history(interval='1m',period='1d')['Close']
		initial_price = hist[-1]
		ret = hist.pct_change()[1:]
		drift = np.mean(ret)*np.sqrt(256)
		volatility = np.std(hist)
		dt = 1/7
		T = 2
		price_paths = []
		for i in range(0, 1000):
			price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)
		# Create the Bokeh plot
		p = figure(title=f"{ticker} Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[0], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[1],line_width=2,color='red')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[2],line_width=2,color='green')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[3], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[4],line_width=2,color='blue')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[5],line_width=2,color ='yellow')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[6], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[7],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[8],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[9], line_width=2)
		html = file_html(p, CDN, f"{ticker} Closing Prices")
		return html
	return render_template('graph-forecast.html')


@app.route('/1y/forecast', methods=['GET', 'POST'])
def graph_forecast_1y():
	if request.method == 'POST':
		ticker = request.values.get("asset").upper()
		ticker_data = yf.Ticker(ticker)
		hist = ticker_data.history(interval='1d',period='1y')['Close']
		initial_price = hist[-1]
		ret = hist.pct_change()[1:]
		drift = np.mean(ret)*np.sqrt(256)
		volatility = np.std(hist)#*np.sqrt(256)
		dt = 1/256
		T = 2
		price_paths = []
		for i in range(0, 100):
			price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)
		# Create the Bokeh plot
		p = figure(title=f"{ticker} Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[0], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[1],line_width=2,color='red')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[2],line_width=2,color='green')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[3], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[4],line_width=2,color='blue')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[5],line_width=2,color ='yellow')
		p.line(np.linspace(0,len(price_paths[0])),price_paths[6], line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[7],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[8],line_width=2)
		p.line(np.linspace(0,len(price_paths[0])),price_paths[9], line_width=2)
		html = file_html(p, CDN, f"{ticker} Closing Prices")
		return html
	return render_template('graph-forecast.html')

@app.route("/tradeview/<asset>")
def tradeview(asset):
	return render_template("tradeview.html",ticker=asset.upper())

@app.route("/admix")
def admix():
	return render_template("admix.html")

@app.route("/stats")
def stats():
	return render_template("admix-two.html")

@app.route("/tree",methods=['GET','POST'])
def tree():
	if request.method == "POST":
		from tree import binomial_tree
		S0 = float(request.values.get('s0')) # initial stock price
		u = float(request.values.get('u'))# up factor
		d = float(request.values.get('d'))# down factor
		p = float(request.values.get('p'))# probability of up move
		n = int(request.values.get('n'))   # number of steps
		tree = pd.DataFrame(binomial_tree(S0, u, d, p, n))
		html = tree.to_html()
		html_table_with_styles = f"""
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
		<h1><a href="/cmc">Back</a></h1>
		<a onclick="this.href='data:text/html;charset=UTF-8,'+encodeURIComponent(document.documentElement.outerHTML)" href="#" download="./downloads/page.html"><div class="download">Download</div></a>
		<h2>Binomial Matrix Simulation</h2>
		{html}
		"""
		return html_table_with_styles
	return render_template("tree-view.html",style="color red;")

from reverse_bs import calculate_stock_price_and_derivatives
@app.route("/reversed-bs",methods=["GET", "POST"])
def rev_bs():
	if request.method == "POST":
		V = request.values.get("V") 
		delta =request.values.get("delta") 
		K = request.values.get("K")
		T = request.values.get("T") 
		r = request.values.get("r") 
		sigma = request.values.get("sigma")
		St, theta, maturity_sensitivity = calculate_stock_price_and_derivatives(V, delta, K, T, r, sigma)
		return render_template("rev-bs.html",St=St,theta=theta,ms=maturity_sensitivity)
	return render_template("rev-bs.html")


@app.route("/dash")
def dash():
	return render_template("dashboard.html")

@app.route('/stoch/forward-rate', methods=['GET','POST'])
def forward_rate():
	return render_template("forward-rate.html")

@app.route('/stoch/price', methods=['GET','POST'])
def get_stoch_price():
	return render_template("stoch.html")

@app.route('/stoch/fs', methods=['GET','POST'])
def get_stoch_fs():
	return render_template("fs.html")

@app.route('/stoch/g(t,r)', methods=['GET','POST'])
def get_stoch_gtr():
	return render_template("gt.html")

@app.route('/stoch/g&f&s', methods=['GET','POST'])
def get_stoch_gfs():
	return render_template("f-g-s.html")

@app.route('/stoch/p(t,r)', methods=['GET','POST'])
def get_stoch_ptr():
	return render_template("p(t,r).html")

@app.route('/stoch/value', methods=['GET','POST'])
def get_stoch_value():
	return render_template("stoch-value.html")

@app.route('/stoch/Ft', methods=['GET','POST'])
def get_stoch_Ft():
	return render_template("Ft.html")

@app.route('/stoch/C(s)', methods=['GET','POST'])
def get_stoch_Cs():
	return render_template("c(s).html")

@app.route('/stoch/S(t,r)', methods=['GET','POST'])
def get_stoch_Str():
	return render_template("s(r,t).html")

@app.route('/stoch/f3', methods=['GET','POST'])
def get_stoch_f3():
	return render_template("f3.html")

@app.route('/stoch/p4', methods=['GET','POST'])
def get_stoch_p3():
	return render_template("p4.html")

@app.route('/stoch/tools', methods=['GET','POST'])
def get_stoch_tools():
	return render_template("tools.html")

@app.route('/stoch/price-expanded', methods=['GET','POST'])
def get_stoch_p_exp():
	return render_template("price_stoch.html")

@app.route('/stoch/price-classic', methods=['GET','POST'])
def stoch_price_classical():
	return render_template("stoch-pricing-two.html")

@app.route('/dV2', methods=['GET','POST'])
def dV2():
	return render_template("dV2.html")

@app.route('/stoch/v(t)', methods=['GET','POST'])
def volume_diff():
	return render_template("v(t).html")


@app.route('/stock-from-call', methods=['GET','POST'])
def stock_from_call():
	return render_template("stock_from_call.html")

@app.route('/stock-from-put', methods=['GET','POST'])
def stock_from_put():
	return render_template("stock_from_put.html")

@app.route('/stoch/yc-stoch-filt', methods=['GET','POST'])
def yc_stoch_filt():
	return render_template("yc-stoch-filt.html")

@app.route('/yc', methods=['GET','POST'])
def yc():
	return render_template("yc.html")

import priceAlgo as pa
@app.route('/PriceAlgo', methods=['GET','POST'])
def price_Algo():
	if request.method=="POST":
		# # Example Usage
		K = float(request.values.get("K"))  # Strike price
		T = float(request.values.get("T"))  # Time to maturity (1 year)
		s = float(request.values.get("s"))
		t = float(request.values.get("t"))
		r = float(request.values.get("r"))  # Risk-free interest rate (5%)
		v = float(request.values.get("v"))  # Volatility (20%)
		St = float(request.values.get("St"))
		dt = float(request.values.get("dt"))

		p = pa.price(St,K,r,v,t,s,T)
		print(p)
		pdt = pa.dPdt(St,K,r,v,t,s,T)*dt
		pdT = pa.dPdT(St,K,r,v,t,s,T)*dt
		pds = pa.dPds(St,K,r,v,t,s,T)*dt

		return render_template("price_algo.html",pds=pds,pdT=pdT,pdt=pdt,p=p)
	return render_template("price_algo.html")


import option_fwd_algo as ofa
@app.route('/option-fwd-algo', methods=['GET','POST'])
def option_fwd_algo():
	if request.method=="POST":
		tf = float(request.values.get("tf"))  # Strike price
		T2 = float(request.values.get("t2"))
		T1 = float(request.values.get("t1"))
		r1 = float(request.values.get("r1"))  # Risk-free interest rate (5%)
		r2 = float(request.values.get("r2"))  # Risk-free interest rate (5%)
		sigma = float(request.values.get("v"))  # Volatility (20%)
		S0 = float(request.values.get("S0"))
		K = float(request.values.get("K"))
		option_type = request.form['option_type']
		
		p = ofa.option_valuation_with_forward_rate(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
		pdt1 = ofa.dPdT1(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
		pdt2 = ofa.dPdT2(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
		pdT1_2 = ofa.dPdT1_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
		pdT2_2 = ofa.dPdT2_2(S0, K, r1, r2, T1, T2, tf, sigma, option_type)
		
		return render_template("option-fwd-algo.html",p=p,pdt1=pdt1,pdt2=pdt2,pdt1_2=pdT1_2,pdt2_2=pdT2_2)
	return render_template("option-fwd-algo.html")

@app.route('/option/probdist',methods=["GET","POST"])
def prodist():
	if request.method=="POST":
				# Black-Scholes parameters
		S0 = float(request.values.get("s0"))    # Initial stock price
		K = float(request.values.get("k"))     # Strike price
		T = float(request.values.get("t"))      # Time to maturity (in years)
		r = float(request.values.get("r"))   # Risk-free rate
		sigma = float(request.values.get("sigma")) # Volatility of the underlying asset
		n_sim = 100_000_000 # Number of simulations

		# Generate random numbers following a normal distribution for asset price simulation
		np.random.seed(42)
		Z = np.random.randn(n_sim)

		# Simulate stock price at maturity using Geometric Brownian Motion
		ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

		# Calculate the payoff of the European call option
		payoff = np.maximum(ST - K, 0)

		# Calculate the discounted expected payoff (option price)
		option_price = np.exp(-r * T) * np.mean(payoff)

		hist, edges = np.histogram(ST, bins=100, density=True)
		p1 = figure(title=f"Simulated Stock Price Distribution at Maturity (T={T} year)",
				x_axis_label="Stock Price", y_axis_label="Probability Density")
		p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="blue", line_color="white", alpha=0.75)
		# Save the stock price distribution plot to an HTML file
		html1 = file_html(p1, CDN)
		# # Plot the distribution of the option payoffs using Bokeh
		hist_payoff, edges_payoff = np.histogram(payoff, bins=100, density=True)
		p2 = figure(title="Distribution of Option Payoffs", 
					x_axis_label="Payoff", y_axis_label="Probability Density")
		p2.quad(top=hist_payoff, bottom=0, left=edges_payoff[:-1], right=edges_payoff[1:], fill_color="green", line_color="white", alpha=0.75)
		html2 = file_html(p2, CDN)
		return f"""{html1}<br>{html2}"""
	return render_template("probdist.html")

@app.route('/stats/binom',methods=['GET','POST'])
def stats_binom():
	from bin_stats_df import bin_stats_df
	if request.method == "POST":
		t = request.values.get('ticker')
		p = request.values.get('perido')
		i = request.values.get('interval')
		df = bin_stats_df(t)
		html = df.to_html()
		html_table_with_styles = f"""
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
		<h1><a href="/cmc">Back</a></h1>
		<a onclick="this.href='data:text/html;charset=UTF-8,'+encodeURIComponent(document.documentElement.outerHTML)" href="#" download="./downloads/page.html"><div class="download">Download</div></a>
		<h2>Binomial Coefficient Matrix </h2>
		{html}
		"""
		return html_table_with_styles
	return render_template('bin_stats.html')

@app.route('/plotly/<ticker>')
def plot_ly(ticker):
	t = yf.Ticker(ticker.upper())
	history=t.history(period="5y")["Close"]
    # Example plot
	fig = px.scatter(history.values,title=f"{ticker}")
	graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('plot-ly.html', graph_json=graph_json)

@app.route('/annualize',methods=['GET','POST'])
def annualize():
	if request.method == 'POST':
		rate = float(request.values.get('rate'))
		n = float(request.values.get('n'))
		forward = float(request.values.get('forward'))
		annualize = (1+rate/n)**n - 1
		period_type = request.values.get("period_type").lower()
		if period_type == 'single':
			result = (1+annualize)**forward - 1
		elif period_type == 'multi':
		# Calculate the forward rate
			rate_shorter = rate / (1 + annualize)
			rate_longer = rate * (1 + annualize)
			T1 = 1 / n
			T2 = T1 + (forward - rate) / rate_longer
			forward_rate = ((1 + rate_longer) ** T2 / (1 + rate_shorter) ** T1) ** (1 / (T2 - T1)) - 1
			result = forward_rate
		return render_template('annualize.html',result=result)
	return render_template('annualize.html')

@app.route('/multicurve',methods=['GET', 'POST'])
def multi_annualize():
	if request.method == 'POST':
		rate_shorter = float(request.values.get('shortr'))  # 3% annualized rate for 1 year
		rate_longer = float(request.values.get('longr'))   # 4% annualized rate for 2 years
		T1 = float(request.values.get('shortt'))  # Time for the shorter rate
		T2 = float(request.values.get('longt'))  # Time for the longer rate
		# Calculate forward rate
		forward_rate = ((1 + rate_longer) ** T2 / (1 + rate_shorter) ** T1) ** (1 / (T2 - T1)) - 1
		print(f"Implied Forward Rate between {T1} and {T2} years: {forward_rate:.2%}")
		return render_template("multicurve.html",rate=forward_rate*100)
	return render_template("multicurve.html")


@app.route('/ddm-spread',methods=["GET","POST"])
def ddm_spread():
	from wacc import Rates
	try:
		if request.method =='POST':
			ticker = request.values.get('ticker').upper()
			rf = float(request.values.get('rf'))
			erp = float(request.values.get('erp'))
			cs = float(request.values.get('cs'))
			price = float(request.values.get('price'))
			interest_coverage = get_interestCoverage(ticker)
			taxes = get_taxes(ticker)
			reg = get_beta(ticker)
			rd = get_costDebt(ticker)
			debt   = get_debt(ticker)
			equity = get_equity(ticker)
			rate = Rates(debt, equity, taxes)
			ke = rate.re(reg, rf, erp, cs)
			costOfCapital = rate.wacc(rd,ke)
			current_price = get_price(ticker) 
			div_growth = implied_div_growth_rate(ke, get_dps(ticker), current_price)
			dps = get_dps(ticker)
			spread = 0
			for i in range(1000):
				result = (dps*(1+div_growth))/(ke - div_growth + spread)
				if result > price:
					spread = spread - .001
				elif result < price:
					spread = spread + .001
				elif (result-price < 1e-6):
					spread = spread
					break
			return render_template("ddm-spread.html",spread=spread*100,name=ticker)
	except IndexError:
		return "INVALID COMPANY STOCK"
	return render_template("ddm-spread.html")



@app.route('/filtration',methods=["GET","POST"])
def filtration():
	from filtration import f
	from scipy.integrate import quad
	if request.method == 'POST':
		target_index = request.values.get('target_index').upper()
		target_stock = request.values.get('target_stock').upper()
		# interval = request.values.get('interval')
		
		# Fetch the index history
		i = yf.Ticker(target_index)
		index = i.history(period='2y')["Close"]
		ret_index = index.pct_change()[1:]

		# Fetch the ticker data from user input
		t = yf.Ticker(target_stock)
		history = t.history(period='2y')
		close = history["Close"]
		ret = close.pct_change()[1:]  # Percentage returns

		# Normalize returns
		norm_ret = [(i - np.mean(ret)) / np.std(ret) for i in ret]
		# Apply f(t) to normalized returns
		r = np.array([f(i) for i in norm_ret])
		# Prepare index returns as features for regression
		X = np.array([ret_index]).T
		X = X.reshape(-1, 1)
		# Fit linear regression model
		model = LinearRegression().fit(X[:], r)
		# Make a prediction for the next day (or any other future time point)
		# Here, we use the last known return value (ret_index[-1]) to predict
		next_day_ret = np.array([ret_index[-1]]).reshape(-1, 1)
		predicted_r = model.predict(next_day_ret)
		pred = (predicted_r + np.mean(ret))*np.std(ret)
		# Variance calculations for price and returns
		var_price = np.var(close)/len(close)
		var_ret = np.var(ret)*np.sqrt(12)
		var_transform = np.sqrt(r@r)
		return render_template("filtration.html",filt=predicted_r[0])
	return render_template("filtration.html")

@app.route("/drift&vol",methods=["GET","POST"])
def drift_vol():
	if request.method == "POST":
		ticker = request.values.get("ticker").upper()
		p = request.values.get("p")
		i = request.values.get("i")
		data = yf.Ticker(ticker).history(period=p,interval=i)#(ticker, start="2020-01-01", end="2023-12-31")
		data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
		# Time step (e.g., daily returns, assume 252 trading days in a year)
		delta_t = 1 / 252
		# Drift (mean of log returns divided by time step)
		mu = data['Log Returns'].mean() / delta_t
		# Volatility (standard deviation of log returns divided by square root of time step)
		sigma = data['Log Returns'].std() / np.sqrt(delta_t)
		return render_template("drift&vol.html",mu=mu,sigma=sigma)
	return render_template("drift&vol.html")

@app.route("/value-algo")
def add_value_algo():
	return render_template("value-algo.html")

@app.route("/p-algo-one")
def palgoone():
	return render_template("p-algo-one.html")

@app.route("/p-algo-two")
def palgotwo():
	return render_template("p-algo-two.html")

@app.route("/dV")
def dV():
	return render_template("dV.html")

@app.route("/v3algo",methods=["GET","POST"])
def value_algo_three():
	return render_template("v3.html")

@app.route("/reverse_bs",methods=["GET","POST"])
def reverse_bs():
	return render_template("reverse_bs.html")

# import reverse_bs_2 as rbs2
@app.route("/reverse_bs2", methods=["GET","POST"])
def reverse_bs2():
	return render_template("reverse_bs2.html")


@app.route("/EqHx", methods=["GET","POST"])
def EqHx():
	return render_template("EqHx.html")

from option_pricing import compute_stock_price_from_option
@app.route("/options/pricing",methods=["GET","POST"])
def options_pricing():
	if request.method == "POST":
		option_price = float(request.form['option_price'])
		K_min = float(request.form['k_min'])
		K_max = float(request.form['k_max'])
		T = float(request.form['maturity_time'])
		sigma2 = float(request.form['variance'])
		r = float(request.form['interest_rate']) # Risk-free rate (5%)
		sigma = np.sqrt(sigma2)
		estimated_stock_price = compute_stock_price_from_option(option_price, T, r, K_min, K_max, sigma)
		print(estimated_stock_price)
		return render_template("options_pricing.html",stock_price=estimated_stock_price)
	return render_template("options_pricing.html")

import bs_diff_two as bdt
@app.route("/bs_diff_2",methods=["GET","POST"])
def bs_diff_2():
	if request.method == "POST":
		option_price = float(request.form['option_price'])
		K = float(request.form['strike_price'])
		T = float(request.form['maturity_time'])
		t = float(request.form['current_time'])
		ts = float(request.form['time_scale'])
		r0 = float(request.form['interest_rate'])
		sigma2 = float(request.form['variance'])
		sigma = np.sqrt(sigma2)

		estimated_stock_price = bdt.estimate_stock_price_bs(option_price, K, T, t, ts, r0, sigma)
		dvdt = bdt.dVdt(option_price, K, T, t, ts, r0, sigma)
		value = bdt.option_value(option_price, K, T, t, ts, r0, sigma)
		value2 = bdt.option_value_dT(option_price, K, T, t, ts, r0, sigma)
		value3 = bdt.option_value_dt(option_price, K, T, t, ts, r0, sigma)
		dvdts = bdt.dVdts(option_price, K, T, t, ts, r0, sigma)
		dvdT = bdt.dVdT(option_price, K, T, t, ts, r0, sigma)
		dvdr0 = bdt.dVdr0(option_price, K, T, t, ts, r0, sigma)
		dvdk = bdt.dVdK(option_price, K, T, t, ts, r0, sigma)
		dvdp = bdt.dVdP(option_price, K, T, t, ts, r0, sigma)
		dvdv = bdt.dVdv(option_price, K, T, t, ts, r0, sigma)
		dvdp2 = bdt.dVdP2(option_price, K, T, t, ts, r0, sigma)
		return render_template("bs_diff_2.html",estimated_stock_price=estimated_stock_price,
						 dvdt=dvdt,value=value,dvdts=dvdts,dvdT=dvdT,dvdr0=dvdr0,dvdk=dvdk,
						 dvdp=dvdp,dvdv=dvdv,dvdp2=dvdp2,value2=value2,value3=value3)
	return render_template("bs_diff_2.html")


@app.route("/paramatize", methods=["GET", "POST"])
def paramatize():
    if request.method == "POST":
        receipt = request.values.get("receipt")
        # Safely retrieve and default values
        mu = request.values.get("mu") or "0"
        sigma = request.values.get("sigma") or "0"
        reversion = request.values.get("reversion") or "0"
        spread = request.values.get("spread") or "0"
        forward = request.values.get("forward") or "0"
        risk_neutral = request.values.get("risk_neutral") or "0"
        filtration = request.values.get("filtration") or "0"
        time_float = request.values.get("time_float") or "0"
        target_price = request.values.get("target_price") or "0"
        delta = request.values.get("delta") or "0"
        rho = request.values.get("rho") or "0"
        theta = request.values.get("theta") or "0"
        vega = request.values.get("vega") or "0"
        dividend_yield = request.values.get("dividend_yield") or "0"
        coe = request.values.get("coe") or "0"
        cod = request.values.get("cod") or "0"
        wacc = request.values.get("wacc") or "0"
        rf = request.values.get("rf") or "0"

        try:
            inv = InvestmentDatabase.query.filter_by(receipt=receipt).first()
            if not inv:
                return "Invalid receipt", 400

            new_token = TokenParameters(
                owner=inv.owner,
                investment_name=inv.investment_name,
                receipt=inv.receipt,
                mu=float(mu),
                rf=float(rf),
                sigma=float(sigma),
                reversion=float(reversion),
                spread=float(spread),
                forward=float(forward),
                risk_neutral=float(risk_neutral),
                filtration=float(filtration),
                time_float=float(time_float),
                target_price=float(target_price),
                delta=float(delta),
                rho=float(rho),
                theta=float(theta),
                vega=float(vega),
                dividend_yield=float(dividend_yield),
                coe=float(coe),
                cod=float(cod),
                wacc=float(wacc),
            )
            db.session.add(new_token)
            db.session.commit()
            return "success"
        except ValueError as e:
            db.session.rollback()
            return f"Invalid input: {e}", 400
        except Exception as e:
            db.session.rollback()
            return f"An error occurred: {e}", 500

    return render_template("parameter-form.html")

@app.route("/test")
def test():
    # Create a sample Plotly figure
	t = yf.Ticker('AAPL').history()["Close"]
	tp = np.linspace(0,len(t),len(t))
	fig = px.line(x=t.index, y=t, title="Interactive Line Chart")

	# Convert figure to JSON
	graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# Pass JSON data to template
	return render_template("test.html",graph_json=graph_json)


@app.route("/kappa",methods=["GET", "POST"])
def kappa():
	if request.method == "POST":
		# Create a sample Plotly figure
		# Define parameters
		ticker = request.values.get("ticker").upper()  # Replace with any stock ticker

		stock_data = yf.Ticker(ticker).history(period='max', interval='1d')

		# Compute log prices
		stock_data["Log Price"] = np.log(stock_data["Close"])

		# Compute log price differences (dX_t)
		stock_data["dX_t"] = stock_data["Log Price"].diff()

		# Compute lagged log price (X_t-1)
		stock_data["Lagged Log Price"] = stock_data["Log Price"].shift(1)

		# Drop NaN values
		stock_data.dropna(inplace=True)

		# Run OLS regression: dX_t = θ(μ - X_t-1) + noise
		X = stock_data["Lagged Log Price"]
		y = stock_data["dX_t"]

		X = sm.add_constant(X)  # Add constant for μ
		model = sm.OLS(y, X).fit()
		theta = -model.params[1]  # Negative coefficient gives mean reversion speed
		mu = model.params[0] / theta  # Long-term mean level

		return render_template("kappa.html", theta=theta, mu=mu)
	return render_template("kappa.html")


@app.route('/token-parameters/<int:id>')
def token_parameters(id):
	try:
		params = TokenParameters.query.get_or_404(id)
		token_data = {
		"owner": params.owner,
		"investment_name": params.investment_name,
		"receipt": params.receipt,
		"mu": float(params.mu),
		"rf": float(params.rf),
		"sigma": float(params.sigma),
		"reversion": float(params.reversion),
		"spread": float(params.spread),
		"forward": float(params.forward),
		"risk_neutral": float(params.risk_neutral),
		"filtration": float(params.filtration),
		"time_float": float(params.time_float),
		"target_price": float(params.target_price),
		"delta": float(params.delta),
		"rho": float(params.rho),
		"theta": float(params.theta),
		"vega": float(params.vega),
		"dividend_yield": float(params.dividend_yield),
		"coe": float(params.coe),
		"cod": float(params.cod),
		"wacc": float(params.wacc)}
		return jsonify(token_data)
	except Exception as e:
		return str(e), 500


schedule.every(5).minutes.do(update)


if __name__ == '__main__':
	with app.app_context():
		db.create_all()
		PendingTransactionDatabase.genisis() 
	def run_scheduler():
		while True:
			with app.app_context():
				schedule.run_pending()
				time.sleep(3600)  #
	schedule_thread = threading.Thread(target=run_scheduler, daemon=True)
	schedule_thread.start()
	app.run(host="0.0.0.0",port=8080)