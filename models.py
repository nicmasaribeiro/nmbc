from flask import Flask, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime
import time
import enum
from flask_bcrypt import Bcrypt
from cryptography.hazmat.primitives.asymmetric import rsa 
from classes import PrivateWallet, Balance
import socket
from sqlalchemy import create_engine
from sqlalchemy.orm import *
from flask import Flask, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sqlalchemy import create_engine, ARRAY
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum
import yfinance as yf
import os
import sys
from sqlalchemy.ext.mutable import MutableList
import datetime as dt


UPLOAD_FOLDER = 'local'
ALLOWED_EXTENSIONS = {'txt', 'html','py','pdf','cpp'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/blockchain.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blockchain.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB limit

# Initialize SQLAlchemy
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
bcrypt = Bcrypt(app)
login_manager.login_view = 'login'

engine = create_engine('sqlite:///blockchain.db')
Session = sessionmaker(bind=engine)


class SharedWalletDB(db.Model):
    __tablename__ = 'shared_wallets'
    
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String, unique=True, nullable=False)
    balance = db.Column(db.Float, default=0)
    password = db.Column(db.String(1024))
    coins = db.Column(db.Float, default=10000)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    coinbase_wallet = db.Column(db.String)
    token = db.Column(db.String(3072))
    swap_debt_balance = db.Column(db.Float, default=0)
    swap_credit_balance = db.Column(db.Float, default=0)

class WalletDB(db.Model):
    __tablename__ = 'wallets'
    
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String, unique=True, nullable=False)
    eth_address = db.Column(db.String, unique=True, nullable=False)
    private_key = db.Column(db.String, unique=True, nullable=False)
    balance = db.Column(db.Float, default=0)
    password = db.Column(db.String(1024))
    coins = db.Column(db.Float, default=10000)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    coinbase_wallet = db.Column(db.String)
    token = db.Column(db.String(3072))
    swap_debt_balance = db.Column(db.Float, default=0)
    swap_credit_balance = db.Column(db.Float, default=0)

    def set_transaction(sender_wallet, recv_wallet, value):
        sender = sender_wallet
        recv = recv_wallet
        money = value
        sender_bal = sender_wallet.balance
        recv_bal = recv_wallet.balance
        if sender_bal > float(value):
            sender_new_bal = float(sender_bal) - float(value)
            recv_new_bal = float(recv_bal) + float(value)
            sender_wallet.balance = sender_new_bal
            recv_wallet.balance = recv_new_bal
            db.session.commit()

    def add_money(self,value):
        self.balance+=float(value)
        db.session.commit()
    
    def add_coins(self,value):
        self.coins+=float(value)
        db.session.commit()
        
    def sell_coins(value):
        Wallet.balance += value
        Wallet.coins -= value
        db.session.commit()
    
    def buy_coins(value):
        Wallet.balance -= value
        Wallet.coins += value
        db.session.commit()


class ForumPost(db.Model):
    __tablename__ = 'forum_posts'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    file_path = db.Column(db.String(512))
    tags = db.Column(db.String, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("Users", backref="forum_posts")
    attachments = db.relationship("ForumAttachment", back_populates="post", lazy=True)  # ✅


class ForumAttachment(db.Model):
    __tablename__ = 'forum_attachments'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filedata = db.Column(db.LargeBinary, nullable=False)  # 🆕 store the file content
    post_id = db.Column(db.Integer, db.ForeignKey("forum_posts.id"), nullable=False)
    post = db.relationship("ForumPost", back_populates="attachments")




class ForumComment(db.Model):
    __tablename__ = 'forum_comments'
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey("forum_posts.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey("forum_comments.id"), nullable=True)  # 🆕
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, default=0)  # 🆕
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    post = db.relationship("ForumPost", backref="comments")
    user = db.relationship("Users", backref="forum_comments")
    replies = db.relationship("ForumComment", backref=db.backref("parent", remote_side=[id]), lazy=True)





class SocialNetwork(db.Model):
    __tablename__ = 'social'
    
    id = db.Column(db.Integer,unique=True, primary_key=True)
    user = db.Column(db.String)
    friend = db.Column(db.String)
    
class BettingHouse(db.Model):
    __tablename__ = 'house'

    id = db.Column(db.Integer,unique=True, primary_key=True)
    balance = db.Column(db.Float, default=0)
    coins = db.Column(db.Float, default=1000000000)
    
    def cash_fee(self,value):
        self.balance +=float(value)
        db.session.commit()
    
    def coin_fee(self,value):
        self.balance += float(value)
        db.session.commit()
    
class Users(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer,unique=True, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120))
    payment_id = db.Column(db.String(1024))
    private_wallet = PrivateWallet()
    personal_token = db.Column(db.String(3072))
    private_token = db.Column(db.String(3072))
    cell_number = db.Column(db.String())
    wallet_id = db.Column(db.Integer, db.ForeignKey('wallets.id'), nullable=True)
    wallet = db.relationship('WalletDB', backref='user', uselist=False)


class NotebookSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    notebook_filename = db.Column(db.String(128))
    score = db.Column(db.Float)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))


class UserNotebook(db.Model):
    published = db.Column(db.Boolean, default=False)
    published_at = db.Column(db.DateTime, default=db.func.now())
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(128))
    content = db.Column(db.Text)  # JSON string (list of cells)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())
    tags = db.Column(db.String, default="")
    is_sequential = db.Column(db.Boolean, default=False)
    is_for_sale = db.Column(db.Boolean, default=False)
    price = db.Column(db.Float, default=0.0)




class DatasetMeta(db.Model):
    __tablename__ = 'dataset_meta'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename = db.Column(db.String, nullable=False)
    description = db.Column(db.String, default="")
    uploaded_at = db.Column(db.DateTime, default=db.func.now())


class TransactionType(enum.Enum):
    send = "send"
    SEND = "SEND"
    receive = "receive"
    internal_wallet = "internal_wallet"
    swap = "swap"
    INVESTMENT = "INVESTMENT"
    investment = "investment"# <-- Add this if needed

class DualFactor(db.Model):
    __tablename__ = 'dual_factor'

    id = db.Column(db.Integer, primary_key=True)
    dual_factor_signature = db.Column(db.String, nullable=False)
    identifier =  db.Column(db.String, nullable=False)
    username = db.Column(db.String)
    from_address = db.Column(db.String, db.ForeignKey('wallets.address'), nullable=False)  # Ensure Not NULL
    to_address = db.Column(db.String, db.ForeignKey('wallets.address'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class TransactionDatabase(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    txid = db.Column(db.String, nullable=False)
    username = db.Column(db.String)
    from_address = db.Column(db.String, db.ForeignKey('wallets.address'), nullable=False)  # Ensure Not NULL
    to_address = db.Column(db.String, db.ForeignKey('wallets.address'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.Enum(TransactionType), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    signature = db.Column(db.String(1024))
    from_wallet = db.relationship('WalletDB', foreign_keys=[from_address])
    to_wallet = db.relationship('WalletDB', foreign_keys=[to_address])

class Blog(db.Model):
    __tablename__ = 'blog'
    
    id = db.Column(db.Integer,unique=True ,primary_key=True)
    title = db.Column(db.String)
    content = db.Column(db.Text)
    f = db.Column(db.LargeBinary)
    thread = db.Column(db.String)

class Notebook(db.Model):
    __tablename__ = 'notebook'
    
    id = db.Column(db.Integer,unique=True ,primary_key=True)
    user = db.Column(db.String)
    title = db.Column(db.String)
    content = db.Column(db.Text)
    f = db.Column(db.LargeBinary)
    file_type = db.Column(db.String)
    thread = db.Column(db.String)
    receipt = db.Column(db.String) 


class Peer(db.Model):
    __tablename__ = 'peers'
    
    id = db.Column(db.Integer,unique=True ,primary_key=True)
    user_address = db.Column(db.String, unique=True, nullable=False)#, unique=True, nullable=False
    pk = db.Column(db.String(120))
    miner_wallet =  db.Column(db.Integer, default=0)
    cash = db.Column(db.Integer, default=0)
    keyPair = db.Column(db.LargeBinary(1024))
    email = db.Column(db.String(120))
    password = db.Column(db.String(120), nullable=False)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    def add_coins(self,value):
        self.miner_wallet += value
        db.session.commit()
        
    def sell_coins(self,value):
        self.miner_wallet -= value
        self.cash += value
        db.session.commit()


class Swap(db.Model):
    __tablename__ = 'swap'

    id = db.Column(db.Integer, primary_key=True)
    notional = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(), nullable=False)
    fixed_rate = db.Column(db.Float, nullable=False)
    floating_rate_spread = db.Column(db.Float, nullable=False)
    equity = db.Column(db.String, nullable=False)
    equity_rate = db.Column(db.Float, default=0)
    amount = db.Column(db.Float, nullable=False)
    total_amount = db.Column(db.Float,default=0)
    fee = .01 #db.Column(db.Float, default=0)
    maturity = db.Column(db.Float, default=0)
    counterparty_a = db.Column(db.String(100), nullable=False)
    counterparty_b = db.Column(db.String(100), nullable=False)
    receipt = db.Column(db.String(100), nullable=False)
    blocks = db.relationship('SwapBlock', backref='swap', lazy=True)

class SwapBlock(db.Model):
    __tablename__ = 'swap_block'

    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    data = db.Column(db.Text, nullable=False)
    status = db.Column(db.String, nullable=True)  # ✅ Allow NULL values
    previous_hash = db.Column(db.String, nullable=False)
    hash = db.Column(db.String, nullable=False)
    swap_id = db.Column(db.Integer, db.ForeignKey('swap.id'), nullable=False)


class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.String(512), nullable=False)
    receipt = db.Column(db.String)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sender = db.relationship('Users', foreign_keys=[sender_id])
    receiver = db.relationship('Users', foreign_keys=[receiver_id])

class SwapTransaction(db.Model):
    __tablename__ = 'swap_transactions'

    id = db.Column(db.Integer, primary_key=True)
    swap_id = db.Column(db.Integer, nullable=False)
    receipt = db.Column(db.String(100), nullable=False)
    sender = db.Column(db.String, nullable=False)
    receiver = db.Column(db.String, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String, default="Pending")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Block(db.Model):
    __tablename__ = 'blocks'
    
    id = db.Column(db.Integer,unique=True,primary_key=True)
    index = db.Column(db.Integer)
    previous_hash = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    hash = db.Column(db.String)
    transactions = db.Column(db.String())

class Chain(db.Model):
    __tablename__ = 'chain'
    
    id = db.Column(db.Integer, primary_key=True)
    txid = db.Column(db.String, nullable=False)
    username = db.Column(db.String)
    from_address = db.Column(db.String, db.ForeignKey('wallets.address'))
    to_address = db.Column(db.String, db.ForeignKey('wallets.address'))
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    type = db.Column(db.Enum(TransactionType), nullable=False)
    signature = db.Column(db.String(1024))
    from_wallet = db.relationship('WalletDB', foreign_keys=[from_address])
    to_wallet = db.relationship('WalletDB', foreign_keys=[to_address])
    
class PendingTransactionDatabase(db.Model):
    __tablename__ = 'pending_transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    txid = db.Column(db.String, nullable=False)
    username = db.Column(db.String)
    from_address = db.Column(db.String, db.ForeignKey('wallets.address'))
    to_address = db.Column(db.String, db.ForeignKey('wallets.address'))
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    type = db.Column(db.Enum(TransactionType), nullable=False)
    signature = db.Column(db.String(1024))
    from_wallet = db.relationship('WalletDB', foreign_keys=[from_address])
    to_wallet = db.relationship('WalletDB', foreign_keys=[to_address])

    def genisis():
        genisis = PendingTransactionDatabase(
            txid=os.urandom(10).hex(),
            username='',
            from_address='',
            to_address='',
            amount=1,
            timestamp=dt.datetime.now(),
            type='investment',
            signature=os.urandom(10).hex(),
        )
        db.session.add(genisis)
        db.session.commit()

class PrivateBlock:
	def __init__(self, index, previous_hash, timestamp, transactions, hash=None):
		self.index = index
		self.previous_hash = previous_hash
		self.timestamp = timestamp
		self.transactions = transactions
		self.hash = hash or self.calculate_hash()
		
	def calculate_hash(self):
		return hashlib.sha256(str(self.index).encode()) .hexdigest()   

class Portfolio(db.Model):
    __tablename__ = 'portfolio'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1024))
    username = db.Column(db.String(1024),unique=False)
    token_name = db.Column(db.String(1024),unique=False)
    token_address = db.Column(db.String(1024), unique=False)
    user_address = db.Column(db.String(1024), unique=False)#, unique=True, nullable=False
    transaction_receipt = db.Column(db.String)
    price = db.Column(db.Float,default=0.0)
    mean = db.Column(db.Integer,default=0.0)
    std = db.Column(db.Float, default=0)
    weight = db.Column(db.Float, default=0)

class AssetToken(db.Model):
    __tablename__ = 'asset_token'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(1024),unique=False)
    token_name = db.Column(db.String(1024),unique=False)
    token_address = db.Column(db.String(1024), unique=False)
    user_address = db.Column(db.String(1024), unique=False)#, unique=True, nullable=False
    transaction_receipt = db.Column(db.String)
    quantity = db.Column(db.Integer,default=0.0)
    cash = db.Column(db.Float, default=0)
    coins = db.Column(db.Float, default=0)

class CoinDB(db.Model):
    __tablename__ = 'coins'
    
    id = db.Column(db.Integer,unique=True, primary_key=True)
    market_cap = db.Column(db.Integer,default=0)
    staked_coins = db.Column(db.Integer,default=0)
    new_coins = db.Column(db.Integer,default=0)
    dollar_value = db.Column(db.Integer,default=0.01)
    total_coins = db.Column(db.Integer,default=1_000_000_000_000)
    
    def gas(self,blockchain,g):
        if 10 > g > 1:
            dif = 10 - g
            chain = blockchain.chain
            for i in chain:
                nonce, hash_result, time_taken = blockchain.proof_of_work(i, difficulty=5)
                self.new_coin(float(time_taken))
            return "Success"
        else:
            return "Wrong Gas"
        
    def new_coin(self,value):
        self.new_coins += value
        db.session.commit()
        return self.new_coins
        
    def proccess_coins(self,blockchain):
        new=[]
        for i in blockchain.stake:
            nonce, hash_result, time_taken = blockchain.proof_of_work(i,5)
            new.append(float(time_taken))
        self.staked_coins = sum(new)
        db.session.commit()
    
    def convert_mc(self):
        new=[]
        for coin in self.staked_coins:
            new.append(coin)
        self.market_cap = sum(new)
        db.session.commit()

class InvestmentType(enum.Enum):
    call = 'call'
    put = 'put'    

class TokenParameters(db.Model):
    __tablename__ = 'parameters'

    id = db.Column(db.Integer, unique=True ,primary_key=True)
    owner =  db.Column(db.String(1024))
    investment_name = db.Column(db.String(1024))
    receipt = db.Column(db.String(1024),unique=True)
    mu = db.Column(db.Float())
    sigma = db.Column(db.Float())
    reversion = db.Column(db.Float())
    spread = db.Column(db.Float())
    forward = db.Column(db.Float())
    rf = db.Column(db.Float())
    risk_neutral = db.Column(db.Float())
    filtration = db.Column(db.Float())
    time_float = db.Column(db.Float())
    target_price = db.Column(db.Float())
    delta = db.Column(db.Float())
    rho = db.Column(db.Float())
    theta = db.Column(db.Float())
    vega = db.Column(db.Float())
    dividend_yield = db.Column(db.Float())
    coe = db.Column(db.Float())
    cod = db.Column(db.Float())
    wacc = db.Column(db.Float())


class InvestmentDatabase(db.Model):
    __tablename__ = 'investments'
    
    id = db.Column(db.Integer, unique=True ,primary_key=True)
    owner =  db.Column(db.String(1024))
    investment_name = db.Column(db.String(1024))
    password = db.Column(db.String(1024))
    quantity = db.Column(db.Float(),default=0.0)
    market_cap = db.Column(db.Float(), default=0.0)
    change_value = db.Column(db.Float(), default=0.0)
    starting_price = db.Column(db.Float(), default=0.0)
    market_price = db.Column(db.Float,default=0.0)
    coins_value = db.Column(db.Float(), default=0.0)
    investment_type = db.Column(db.Enum(InvestmentType))
    investors = db.Column(db.Integer)
    receipt = db.Column(db.String(1024),unique=True)
    risk_neutral = db.Column(db.Float)
    spread = db.Column(db.Float)
    reversion = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    time_float = db.Column(db.Float())
    target_price = db.Column(db.Float())
    stoch_price = db.Column(db.Float())
    tokenized_price = db.Column(db.Float,default=0.0) # tokenized_value
    ls = MutableList()

    def update_token_value(self):
        self.tokenized_price = self.market_cap/self.coins_value
        db.session.commit()
    
    def add_market_cap(self,value):#name
        self.market_cap+=float(value)
        db.session.commit()
    
    def add_stake(self,value):#name
        self.coins_value+=float(value)
        db.session.commit()
    
    def add_investor(self):
        self.investors += 1
        db.session.commit()
   
    def append_investor_token(self,name,address,receipt,amount,currency):
        self.ls += [{'name':name,
            'address':address,
            'receipt':receipt,
            'amount':amount,
            'currency':currency}]
        db.session.commit()

class ValuationDatabase(db.Model):
    __tablename__ = 'valuation'
    
    id = db.Column(db.Integer, unique=True ,primary_key=True)
    owner =  db.Column(db.String(1024))
    target_company = db.Column(db.String(1024))
    forecast = db.Column(db.Float(),default=0.0)
    wacc = db.Column(db.Float(), default=0.0)
    roe  = db.Column(db.Float(), default=0.0)
    rd = db.Column(db.Float(), default=0.0)
    change_value = db.Column(db.Float(), default=0.0)
    price = db.Column(db.Float(),default=1)
    receipt = db.Column(db.String(),unique=True)
    valuation_model = db.Column(db.LargeBinary()) # tokenized_value



class ValuationType(enum.Enum):
    dcf = 'dcf'
    ddm = 'ddm'
    optimization = 'optimization'
    
class GeneralValuationDatabase(db.Model):
    __tablename__ = 'general_valuation'
    
    id = db.Column(db.Integer, unique=True ,primary_key=True)
    owner =  db.Column(db.String(1024))
    target_company = db.Column(db.String(1024))
    forecast = db.Column(db.Float(),default=0.0)
    expected_change = db.Column(db.Float(),default=0.0)
    receipt = db.Column(db.String(),unique=True)
    type = db.Column(db.Enum(ValuationType), nullable=False)
    valuation_model = db.Column(db.LargeBinary())


class OptimizationToken(db.Model):
    __tablename__ = 'optimization_token'
    
    id = db.Column(db.Integer, primary_key=True)
    file_data = db.Column(db.LargeBinary, nullable=False)  # Store the file as binary (BLOB)
    receipt = db.Column(db.String)
    modified_data = db.Column(db.LargeBinary, nullable=False)
    input_data = db.Column(db.LargeBinary, nullable=False)
    grade = db.Column(db.Integer, default=0)
    output_data = db.Column(db.LargeBinary, nullable=False)
    additional_data = db.Column(db.LargeBinary, nullable=False)
    string_data = db.Column(db.Text(), nullable=False)
    filename = db.Column(db.String(), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    description = db.Column(db.String())


class Optimization(db.Model):
    __tablename__ = 'optimization'
    
    id = db.Column(db.Integer, primary_key=True)
    file_data = db.Column(db.LargeBinary, nullable=False)  # Store the file as binary (BLOB)
    description = db.Column(db.String)
    receipt = db.Column(db.String)
    filename = db.Column(db.String(), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    price = db.Column(db.Float)
    input_data = db.Column(db.LargeBinary, nullable=False)


class TrackInvestors(db.Model):
    __tablename__ = 'tracking'
    
    id = db.Column(db.Integer, unique=True ,primary_key=True)
    receipt = db.Column(db.String(1024),unique=False)
    tokenized_price = db.Column(db.Float,default=0.0)
    owner =  db.Column(db.String(1024))
    investment_name = db.Column(db.String(1024))
    investor_name = db.Column(db.String(1024))
    investor_token = db.Column(db.String(1024))
    
    
class Network:
    def __init__(self):
        self.pending_transactions = []
        self.approved_transactions = []
        self.stake = []
        self.chain = [self.create_genesis_block()]
        self.senders = []
        self.money = []
        self.receipts = []
        self.market_cap = 0.0001
        
    def set_market_cap(self, value):
        self.market_cap = value
        
    def add_transaction(self,transaction):
        self.pending_transactions.append(transaction)
    
    def create_genesis_block(self):
        return PrivateBlock(0, "0",time.time(), [], "0")
   
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
            
            for transaction in current_block.transactions:
                if not transaction.is_valid():
                    return False
                
        return True
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def sign_packet(self,packet:bytes, key):
        hash = int.from_bytes(sha512(packet).digest(), byteorder='big')
        signature = pow(hash, key.d, key.n)
        print("Signature:", hex(signature))
        self.receipts.append(signature)
        return (signature,hex(signature))
    
    def verify_packet(self,packet:bytes, key, signature):
        hash = int.from_bytes(sha512(packet).digest(), byteorder='big')
        hashFromSignature = pow(signature, key.e, key.n)
        print("Signature valid:", hash == hashFromSignature)
        return hash == hashFromSignature
    
    def get_stake(self):
        return self.stake
    
    def get_pending(self):
        return self.pending_transactions
    
    def get_approved(self):
        return self.approved_transactions
    
    def set_transaction(self, sender_wallet, recv_wallet, value, blockchain):
        sender_user = sender_wallet.address
        recv_public_key = recv_wallet.address
        money = value
        bal = sender_wallet.balance
        new_bal = float(bal) - float(value)
        sender_wallet.balance = new_bal
        db.session.commit()
    
    def process_transaction(self, sender_wallet, recv_wallet, value, index, coin, blockchain):
        pending = blockchain.pending_transactions
        r =  {"id":os.urandom(10),"pending":[pending]}
        blockchain.receipts.append(r)
        trans = blockchain.pending_transactions[index]
        blockchain.approved_transactions.append(trans)
        blockchain.pending_transactions.pop(index)
        result = coin.stake_coins(blockchain.approved_transactions,blockchain.pending_transactions)
        blockchain.stake.append(result)
        gained_coins = sender_wallet.coins + result
        print("gained coins", gained_coins)
        coin.market_cap += gained_coins
        blockchain.market_cap += gained_coins
        return gained_coins
    
    def get_transaction(self, sender_wallet, recv_wallet, value):
        if sender_wallet.balance <= float(value):
            bal = recv_wallet.balance #private_wallet.get_settled_cash()
            new_bal = bal + float(value)
            recv_wallet.balance = new_bal
            db.session.commit()
        else:
            bal = sender_wallet.balance
            new_bal = bal + float(value)
            sender_wallet.balance = new_bal
            db.session.commit()
   
    def proof_of_work(self,block_data, difficulty=5):
        nonce = 0
        start_time = time.time()
        prefix = '0' * difficulty
        while True:
            nonce += 1
            text = str(block_data) + str(nonce)
            hash_result = hashlib.sha256(text.encode()).hexdigest()
            if hash_result.startswith(prefix):
                end_time = time.time()
                time_taken = end_time - start_time
                return nonce, hash_result, time_taken
    
    def generate_key_pair(self):
        keyPair = rsa.generate_private_key(3,10)
        return keyPair


class Blockchain(Network):
    def __init__(self):
        super(Network).__init__()
        self.market_cap = 0.0001
        self.staked_coins = []
        self.new_coins = 0
        self.dollar_value = 0
        self.chain = [self.create_genesis_block()]
        self.receipts = {"to":[0],"from":[0],"value":[0],'txid':[0]}
        self.approved_transactions = []
        self.pending_transactions = []
        self.money = []
        self.stake = []
        self.difficulty = 5
        self.mining_reward = 100
        self.packets = []
        
    def process_receipts(self,receipts):
        total_sum = 0
        while True:
            once = os.urandom(10).hex() 
            print(once)
            if once.startswith('00') or once.endswith('00'):
                total_sum += self.receipts['value']
                self.stake.append(total_sum)
        return total_sum, once
    
    def add_receipt(self,_to,_from,value,txid):
        self.receipts['to'].append(_to)
        self.receipts['from'].append(_from)
        self.receipts['value'].append(value)
        self.receipts['txid'].append(txid)

    def get_pending(self):
        return self.pending_transactions
    
    def get_approved(self):
        return self.approved_transactions
    
    def create_genesis_block(self):
        return PrivateBlock(0, "0",time.time(), [], "0")
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def generate_key_pair(self):
        keyPair = rsa.generate_private_key(3, 10)
        return keyPair
    
    def sign_packet(self,packet:bytes, key):
        from hashlib import sha512
        hash = int.from_bytes(sha512(packet).digest(), byteorder='big')
        signature = pow(hash, key.d, key.n)
        print("Signature:", hex(signature))
        self.packets.append(signature)
        return (signature,hex(signature))
    
    def verify_packet(self,packet:bytes, key,signature):
        hash = int.from_bytes(sha512(packet).digest(), byteorder='big')
        hashFromSignature = pow(signature, key.e, key.n)
        print("Signature valid:", hash == hashFromSignature)
        return hash == hashFromSignature
    
    def calculate_hash(self):
        return hashlib.sha256(str(self.get_latest_block()).encode()) .hexdigest() 
    
    def proof_of_work(self,block_data, difficulty=5):
        nonce = 0
        start_time = time.time()
        prefix = '0' * difficulty
        while True:
            nonce += 1
            text = str(block_data) + str(nonce)
            hash_result = hashlib.sha256(text.encode()).hexdigest()
            if hash_result.startswith(prefix):
                end_time = time.time()
                time_taken = end_time - start_time
                return nonce, hash_result, time_taken

    
    def mine_pending_transactions(self, mining_reward_address):
        reward_tx = (None, mining_reward_address, self.mining_reward)
        self.pending_transactions.append(reward_tx)
        block = PrivateBlock(len(self.chain), self.get_latest_block().hash, int(time.time()), self.pending_transactions)
        block.hash = block.calculate_hash()  # Simple hash assignment
        self.chain.append(block)
        self.pending_transactions.clear()
        
    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)
        
    def add_block(self,block):
        self.chain.append(block)
        
    def get_balance_of_address(self, address):
        balance = 0
        
        for block in self.chain:
            for trans in block.transactions:
                if trans.from_address == address:
                    balance -= trans.amount
                    
                if trans.to_address == address:
                    balance += trans.amount
                    
        return balance
    
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.hash != self.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
            
            for transaction in current_block.transactions:
                if not transaction.is_valid():
                    return False
                
        return True
    
    
class Coin:
    def __init__(self):
        self.market_cap = 0.0001
        self.staked_coins = []
        self.new_coins = 0
        self.dollar_value = self.stake_coins/self.market_cap
        
    def process_coins(self,blockchain):
        blockchain.mine
        self.new_coins += 1
        return self.new_coins
    
    def stake_coins(self, approved_transactions, pending_transactions,blockchain):
        v = self.process_coins(blockchain)
        len1 = sum(pending_transactions)
        len2 = sum(approved_transactions)
        u = (float(len1) + float(len2)) / float(v)
        return u

import hashlib
import time
class Validator():
    def __init__(self):
        super().__init__()
        self.receipt_hash = []
        self.receipt = []
        self.ledger = {}
        self.ledger_hash = {}
        
    def mine_block(self, net, sender, recv, value, index, c):
        staked_coins = net.get_market_cap()
        earned_coins = net.process_transaction(sender, recv, value, index, c)
        c.market_cap += staked_coins + earned_coins
        self.ledger[sender.get_username()] = earned_coins
        self.receipt.append(earned_coins)
        return earned_coins
    
    def process_receipts(self):
        while True:
            once = os.urandom(10).hex() 
            print(once)
            if once.startswith('00') or once.endswith('00'):
                break
        total_sum = sum(self.receipt)
        self.stake += total_sum
        self.receipt.clear()
        return total_sum, once
    
    def hashing_double(self, value):
        hashed_data = hashlib.sha256(value).hexdigest()
        return hashed_data#int.from_bytes(self.receipt_hash.update(str(value).encode()).digest(), byteorder='big')
    
    def proof_of_work(block_data, difficulty=5):

        nonce = 0
        start_time = time.time()
        prefix = '0' * difficulty
        while True:
            nonce += 1
            text = str(block_data) + str(nonce)
            hash_result = hashlib.sha256(text.encode()).hexdigest()
            if hash_result.startswith(prefix):
                end_time = time.time()
                time_taken = end_time - start_time
                return nonce, hash_result, time_taken
class ProofOfBurn:
    def __init__(self):
        self.burn_address = "0x0000000000000000000000000000000000000000"
        self.burn_records = {}
    
    def generate_burn_hash(self, user, amount):
        """
        Generate a unique hash for the burn transaction.
        """
        data = f"{user}:{amount}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def burn_tokens(self, user, amount):
        """
        Simulate burning tokens by recording the burn transaction.
        """
        burn_hash = self.generate_burn_hash(user, amount)
        timestamp = time.time()
        self.burn_records[burn_hash] = {
            'user': user,
            'amount': amount,
            'timestamp': timestamp,
            'burn_address': self.burn_address
        }
        return burn_hash, timestamp
    
    def verify_burn(self, burn_hash):
        """
        Verify if a burn transaction exists.
        """
        if burn_hash in self.burn_records:
            return True, self.burn_records[burn_hash]
        else:
            return False, None

        