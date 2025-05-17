from flask import Blueprint, request, jsonify,render_template
from web3 import Web3
import os
import json
from dotenv import load_dotenv

load_dotenv()

nativecoin_bp = Blueprint("nativecoin", __name__)

# Load environment variables
PRIVATE_KEY = '0x8601c63DC9Feee842b3c91FD1919304b0e43cA74' #os.getenv("PRIVATE_KEY")
INFURA_URL = 'https://eth-mainnet.g.alchemy.com/v2/iNlxQhyVsGrSkne4fFENpLnkUS-GkhQc'#'http://127.0.0.1:8545/' #'https://eth-sepolia.g.alchemy.com/v2/iNlxQhyVsGrSkne4fFENpLnkUS-GkhQc' #os.getenv("INFURA_URL")
TOKEN_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512" #Web3.to_checksum_address()

# Load ABI
with open("ethereum/artifacts/contracts/NmbcCoin.sol/NativeCoin.json") as f:
    TOKEN_ABI = json.load(f)['abi']
print(TOKEN_ABI)
# Setup Web3
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
contract = w3.eth.contract(address=TOKEN_ADDRESS, abi=TOKEN_ABI)


@nativecoin_bp.route("/ntc/balance/<address>", methods=["GET"])
def get_balance(address):
    try:
        address = Web3.to_checksum_address(address)
        balance = contract.functions.balanceOf(address).call()
        human_balance = w3.from_wei(balance, "ether")
        return jsonify({"address": address, "balance": str(human_balance)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@nativecoin_bp.route("/ntc/transfer", methods=["POST"])
def transfer():
    try:
        data = request.json
        to_address = Web3.to_checksum_address(data["to"])
        amount = float(data["amount"])

        sender = w3.eth.account.from_key(PRIVATE_KEY)
        nonce = w3.eth.get_transaction_count(sender.address)
        tx = contract.functions.transfer(to_address, w3.to_wei(amount, "ether")).build_transaction({
            'from': sender.address,
            'nonce': nonce,
            'gas': 100000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        return jsonify({"tx_hash": w3.to_hex(tx_hash)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@nativecoin_bp.route("/ntc/supply", methods=["GET"])
def get_supply():
    try:
        total = contract.functions.totalSupply().call()
        return jsonify({"total_supply": str(w3.from_wei(total, "ether"))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
