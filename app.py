from flask import Flask, render_template, request, redirect, abort, jsonify,sessions, Response, url_for,send_file,render_template_string,flash
from flask import Blueprint
import asyncio
import socket
import os
from quart import Quart
from web3 import Web3
import os
import csv
import random
import subprocess as sp
import xml.etree.ElementTree as ET
import xml.dom.minidom
import yfinance
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.embed import file_html
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
from cdp import *

api_key = "organizations/5eb0bfbc-3029-4b75-aac6-39ba188d3ac5/apiKeys/ee424a62-beb9-4673-9ef6-7abf2af0d612"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIIrFOL9aVS7DRHGkY8/vyuDIDdW8JBeNf6oraa5c7riOoAoGCCqGSM49\nAwEHoUQDQgAEL5Vod5wi+tHXRmn7aiwwnd12d8brinhlrQsk1nJmeQEC8JpFqAJ+\nTmPiJ3r00ZG3UFuJbGsip9Yia1F+4nAiEQ==\n-----END EC PRIVATE KEY-----\n"
Cdp.configure(api_key, api_secret)
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
p2p = P2PNode('0.0.0.0',PORT)



def recalculate():
	invests = InvestmentDatabase.query.all()
	for i in invests:
		try:
			t = yf.Ticker(i.investment_name.upper())
			prices_vector = t.history(period='5d',interval='1m')
			price = t.history()['Close'][-1]
			s = stoch_price(1/12, i.time_float, i.risk_neutral, i.spread, i.reversion, price, i.target_price)
			i.stoch_price = s
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
	
def update():
	recalculate()
	change_value_update()
	invests = InvestmentDatabase.query.all()
	try:
		for i in invests:
			t = yf.Ticker(i.investment_name.upper())
			prices_vector = t.history(period='5d',interval='1m')
			price = t.history(period='1d',interval='1m')['Close'].iloc[-1]
			i.market_price = price
			db.session.commit()
			current_time = datetime.utcnow()
			time_difference = current_time - i.timestamp
			i.time_float -= time_difference.total_seconds() / (365.25 * 24 * 3600)
			db.commit()
			token_price = black_scholes(price, i.target_price, i.time_float, .05, np.std(t.history(period='1d',interval='1m')['Close'])*np.sqrt(525960),i.investment_type) + derivative_price(prices_vector, i.risk_neutral, i.reversion, i.spread)
			i.tokenized_price = token_price
			i.coins = token_price
			db.session.commit()
	except:
		pass
	return 0


def background_task():
	while True:
		update()	# Do some work here, like checking a database or updating something
		
		# Start the background task in a separate thread
def start_background_task():
	thread = threading.Thread(target=background_task)
	thread.daemon = True  # Ensures the thread exits when the main program does
	thread.start()

def blockchain_broadcast():
	p2p.start_server()

def start_blockchain_broadcast():
	thread = threading.Thread(target=blockchain_broadcast)
	thread.daemon = True  # Ensures the thread exits when the main program does
	thread.start()

@app.route('/task')
def task():
	p2p.connect_to_peer('0.0.0.0',PORT)
	return "Success"

# @app.route("/broadcast")
# def broad():
# 	return Response(node_bc.register_node(f"0.0.0.0:{PORT}"))

# @app.route("/capture")
# def capture():
# 	if request.method == "POST":
# 		t = Transaction(
# 			v = request.values.get("value"),
# 		_from = request.values.get("from"),
# 		_to = request.values.get("to"),
# 		signature = os.urandom(10).hex())
# 		node_bc.broadcast_transaction(t)
# 		node_bc.add_block(node_bc.get_last_block().proof)
# 	return f"Success"

# @app.route("/resolve")
# def resolve_conflicts():
# 	node_bc.resolve_conflicts()
# 	return "Success"

# @app.route("/active/nodes")
# def nodes():
# 	ls = p2p.peers
# 	return f"Success {ls}"


# @app.route("/slash")
# def slash():
# 	HOST = "192.168.1.237"  # The server's hostname or IP address
# 	PORT = 8000  # The port used by the server
# 	url = 'ws://'+ '0.0.0.0' +':'+ str(PORT)
# 	async def run():
# 		web = ws.WebSocket()
# 		await web.connect(url)
# 		await web.recv(1024)
# 		await web.send("Hello")
# 		return web.close()
# 	run()
# 	return "ASYNC"

# 	async def start():
# 		loop = asyncio.get_event_loop()
# 		while True:
# 	#       loop.create_server(('0.0.0.0',PORT))
# 			name = s.getsockname()
# 	#       await s.send(f"{name}".encode())
# 			i = input("==>\t")
# 			await s.sendall(i.encode())
# 			data = s.recv(1024)
# 			print(f"Received {data!r}")
# 			return f"Received {data!r}"


@login_manager.user_loader
def load_user(user_id):
	update()
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
	return render_template('base.html')

@app.route('/signup', methods=['POST','GET'])
def signup():
	if request.method =="POST":
		password = request.values.get("password")
		username = request.values.get("username")
		email = request.values.get("email")
		unique_address = os.urandom(10).hex()
		hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
		new_user = Users(username=username, email=email, 
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
				cb_wallet = Wallet.create()
				cb_data = cb_wallet.export_data()
				data = str(cb_data.to_dict())
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
#	new_transaction = TransactionDatabase()
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

@app.route('/coinbase/<user>', methods=['GET'])
def coinbase(user):
	wallet = WalletDB.query.filter_by(address=user).first()
	coinbase = str(wallet.coinbase_wallet)
	json_string = coinbase.replace("'", '"')
	data = json.loads(json_string)
	wallet_id = data['wallet_id']
	fetched_wallet = Wallet.fetch(wallet_id)
	faucet_transaction = fetched_wallet.faucet()
	if faucet_transaction is None:
		return "Failed to fetch coinbase"
	print(f"Faucet transaction successfully completed: {faucet_transaction}")
	return f"Successfully fetched coinbase {faucet_transaction.transaction_hash}"

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

@login_required
@app.route('/cmc',methods=['GET'])
def cmc():
	user = current_user
	return render_template("cmc.html",user=user)

@login_required
@app.route('/html/trans',methods=['GET'])
def html_trans_database():
	t = TransactionDatabase.query.all()
	return render_template("html-trans.html", trans=t)

@login_required
@app.route('/html/investment/ledger',methods=['GET'])
def html_investment_ledger():
	update()
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


@app.route('/holdings', methods=['GET'])
@login_required
def get_user_wallet():
	update()
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
			profit_loss+=invs.change_value
			total_investments+=invs.tokenized_price*asset.quantity
    
    # Convert the dictionary to a pandas DataFrame
	dataframe = pd.DataFrame(df)
		
    # Transport list for potential future JSON response
	transports_list = [{"address": wallet.address, "balance": wallet.balance, "coins": wallet.coins}]
    
		# Convert DataFrame to HTML table with styles
	html = dataframe.to_html(index=False)
	html_table_with_styles = f"""
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
	return render_template("mywallet.html",wallet=wallet)


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

@app.route('/create/investment', methods=['GET', 'POST'])
def buy_or_sell():
	from pricing_algo import derivative_price
	from bs import black_scholes
	from algo import stoch_price
	from scipy.stats import norm
	def normal_pdf(x, mean=0, std_dev=1):
		return norm.pdf(x, loc=mean, scale=std_dev)
	def C(s):
		K = s**3-3*s**2*(1-s)
		return (s*(s-1/K))**(1/s)*normal_pdf(s)
	update()
	if request.method == "POST":
		user = request.values.get('name')
		invest_name = request.values.get('ticker').upper()
		coins = float(request.values.get('coins'))
		password = request.values.get('password')
		qt = float(request.values.get("qt"))
		target_price = float(request.values.get("target_price"))
		maturity = float(request.values.get("maturity"))
		risk_neutral = float(request.values.get("eta"))
		spread = float(request.values.get("sigma"))
		reversion = float(request.values.get("mu"))
		option_type = request.values.get("option_type").lower()
		user_db = Users.query.filter_by(username=user).first()
		
		if coins < .1:
			return "<h3>Invalid Leverage Value</h3>"
		
		if risk_neutral > 1.1 or risk_neutral < 0 :
			return "Wrong Neutral Measure"
		
		if not user_db:
			return "<h3>User not found</h3>"
		
		ticker = yf.Ticker(invest_name)
		history = ticker.history(period='1d', interval='1m')
		
		if history.empty:
			return "<h3>Invalid ticker symbol</h3>"
		
		price = history['Close'][-1]
		option = black_scholes(price, target_price, maturity, .05, np.std(history['Close'].pct_change()[1:])*np.sqrt(525960),option_type)
		
		def sech(x):
			return 1 / np.cosh(x)
		Px = lambda t: np.exp(-t)*np.sqrt(((t**3-3*t**2*(1-t))*(1-((t**3-3*t**2*(1-t))/sech(t))))**2)
		
		stoch = abs(stoch_price(.1, maturity*12, risk_neutral, spread, reversion, price, target_price,option_type))
		
		token_price = max(0, option + derivative_price(history['Close'], risk_neutral ,reversion, spread)) + C(coins)
		
		
		wal = WalletDB.query.filter_by(address=user).first()
		if wal and wal.coins >= coins:
			receipt = os.urandom(10).hex()
			new_transaction = TransactionDatabase(
				txid=receipt,
				from_address=user_db.personal_token,
				to_address=invest_name,
				amount=coins * qt,
				type='investment',
				username = user,
				signature=sha256(str(user_db.private_token).encode()).hexdigest()
			)
			db.session.add(new_transaction)
			db.session.commit()
			
			new_asset_token = AssetToken(
				username=user,
				token_address=receipt,
				user_address=user_db.personal_token,
				token_name = invest_name,
				transaction_receipt=os.urandom(10).hex(),
				quantity=qt,
				cash = qt * token_price,
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
				spread =  spread,
				reversion=reversion,
				market_cap=qt * price,
				target_price=target_price,
				investment_type=option_type,
				starting_price=price,
				market_price=price,
				timestamp = dt.datetime.utcnow(),
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
			
			# Create the block data
			pen_trans = PendingTransactionDatabase.query.all()[-1]
			all_pending = PendingTransactionDatabase.query.all()
#			
			new_transaction = PendingTransactionDatabase(
				txid=os.urandom(10).hex(),
				username=user,
				from_address=user,
				to_address='market',
				amount=token_price,
				timestamp=dt.datetime.utcnow(),
				type='investment',
				signature=receipt)
			db.session.add(new_transaction)
			db.session.commit()
			
			packet = {
				'index': len(blockchain.chain) + 1,
				'previous_hash': sha256(str(blockchain.get_latest_block()).encode()).hexdigest(),
				'datetime': str(dt.datetime.now()),
				'transactions': all_pending,
			}
			encoded_packet = str(packet).encode().hex()
			blockdata = Block(
				index = len(Block.query.all())+1,
				previous_hash=pen_trans.signature,
				timestamp=dt.datetime.now(),
				hash = encoded_packet,
				transactions = str(all_pending))
			
			db.session.add(blockdata)
			db.session.commit()
			blockchain.add_block(packet)
			node_bc.add_block(packet)
			node_bc.broadcast_block(packet)
			return """<a href='/'><h1>Home</h1></a><h3>Success</h3>"""
		else:
			return "<h3>Insufficient coins in wallet</h3>"
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
	update()
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


@app.route('/asset/info/<int:id>')
def info_assets(id):
	update()
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

@app.route("/left/handshake")
def l_handshake():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Bind before listen
        s.bind(("0.0.0.0", 4001))
        s.listen()

        # Accept the connection
        conn, addr = s.accept()
        with conn:
            key = conn.recv(1024).decode("utf8")
            return f"Received key: {key}"
    except Exception as e:
        return f"Error occurred: {str(e)}"
    finally:
        s.close()

@app.route("/right/handshake")
def r_handshake():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        s.connect(("127.0.0.1", 4001))  # use "127.0.0.1" instead of "0.0.0.0"
        key = os.urandom(10).hex()
        s.send(key.encode("utf8"))
        return f"Sent key: {key}"
    except Exception as e:
        return f"Error occurred: {str(e)}"
    finally:
        s.close()

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
	update()
	if request.method =="POST":
		update()
		address = request.values.get('address')
		user = request.values.get('user')
		password = request.values.get('password')
		invest = InvestmentDatabase.query.filter_by(receipt=address).first()
		wal = WalletDB.query.filter_by(address=user).first()
		user_db = Users.query.filter_by(username=user).first()
		user_token = user_db.personal_token 
		asset_token = AssetToken.query.filter_by(transaction_receipt = address).first()
		if (asset_token != None) and (invest.investors > 1) and (invest != None):
				update()
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

@app.route("/risk-neutral-measure",methods=['GET','POST'])
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

@app.route("/write/blog")
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
	file = request.files['file']
	data = file.read()
	blog = Blog(title=title,content=content,f=data)
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
		change=float(request.values.get("change"))
		price = float(request.values.get("price"))
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
			change_value=0,
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
		wal2 = User.query.filter_by(username=val.owner).first()
		if val.price <= wal.coins:
			wal-= val.price
			wal2+= val.price
			db.session.commit()
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
		ticker = request.values.get("ticker")
		price = request.values.get("price")
		file_data = file.read()
		pending = PendingTransactionDatabase(
			txid=os.urandom(10).hex(),
			username = user.username,
			from_address = user.personal_token,
			to_address = "Valuation Chain",
			amount = price,
			timestamp = dt.date.today(),
			type = 'info-exchange',
			signature = receipt
		)
		db.session.add(pending)
		db.session.commit()
		optimization = Optimization(
							   price=price,
							   filename=name,
							   target=ticker,
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
		ticker = request.values.get("ticker")
		score = float(request.values.get("score"))
		optimal_value = float(request.values.get("optimal_value"))
		# Ensure the receipt exists in the query
		optmimization = Optimization.query.filter_by(receipt=receipt).first()
		if not optmimization:
			return """<h2>Receipt not found</h2>""", 400
		f = request.files['file']
		output_data = f.read()
		token = OptimizationToken(
							file_data=optmimization.file_data,
							target = ticker,
							score = score,
							optimal_value=optimal_value,
							receipt=receipt,
							output_data=output_data,
							filename=optmimization.filename,
							created_at=dt.datetime.now())
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
		ticker = request.values.get("tickers")
		period = request.values.get("period")
		interval = request.values.get("interval")
		
		t = yf.Ticker(ticker)
		history = t.history(period='max',interval='1d')
		prices = history["Close"]
		log_returns = np.log(prices / prices.shift(1)).dropna()
		lagged_returns = log_returns.shift(1).dropna()
		log_returns = log_returns.iloc[1:]
		lagged_returns = lagged_returns.iloc[1:]
		X = lagged_returns.values.reshape(-1, 1)  # Reshape for sklearn
		y = log_returns.values
		lr = LinearRegression()
		fit = lr.fit(X, y[1:])
		beta = fit.coef_
		reversion_coefficient = -np.log(1 - beta)
		
		return render_template('single-reversion.html',reversion=reversion_coefficient.item())
	return render_template('single-reversion.html')

@app.route('/ddm',methods=["GET","POST"])
def ddm():
	from wacc import Rates
	if request.method =='POST':
		ticker = request.values.get('ticker')
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
		return f"<h1>{ddm}</h1><h1>{div_growth}</h1>"
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
		return render_template("mu-sigma.html",rate=pred.item()*100,score=score,price=price)
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
		volatility = np.std(hist)#*np.sqrt(256)
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

@app.route('/stoch/v(t)', methods=['GET','POST'])
def volume_diff():
	return render_template("v(t).html")


@app.route('/stoch/yc-stoch-filt', methods=['GET','POST'])
def yc_stoch_filt():
	return render_template("yc-stoch-filt.html")

@app.route('/yc', methods=['GET','POST'])
def yc():
	return render_template("yc.html")

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
	from bin_stats_df import bin_stats_df#(ticker, period, interval)
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

			
if __name__ == '__main__':
	with app.app_context():
		db.create_all()
		# PendingTransactionDatabase.genisis()
	start_blockchain_broadcast()
	start_background_task()
	app.run(host="0.0.0.0",port=1000)