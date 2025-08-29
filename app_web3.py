# app_web3.py
import os, json
from dotenv import load_dotenv
from flask import Flask, jsonify, request, abort
from web3 import Web3, HTTPProvider

load_dotenv()
app = Flask(__name__)

HTTP_URL = os.getenv("WEB3_HTTP_URL")
CHAIN_ID = int(os.getenv("CHAIN_ID", "1"))
CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))

# Initialize Web3
w3 = Web3(HTTPProvider(HTTP_URL))
assert w3.is_connected(), "Web3 provider not connected"

# Load ABI (place abi.json next to this file)
with open("abi.json") as f:
    ABI = json.load(f)
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

@app.get("/api/chain")
def chain_info():
    return jsonify({
        "connected": w3.is_connected(),
        "chain_id": w3.eth.chain_id,
        "latest_block": w3.eth.block_number
    })

@app.get("/api/contract/reads")
def contract_reads():
    # Example: public view reads
    try:
        symbol = contract.functions.symbol().call()
        total_supply = contract.functions.totalSupply().call()
        return jsonify({"symbol": symbol, "totalSupply": str(total_supply)})
    except Exception as e:
        return abort(400, str(e))

@app.post("/api/tx/prepare")
def prepare_tx():
    """
    Build a transaction payload that the FRONTEND will sign with MetaMask.
    Body: {"from":"0x...", "method":"transfer", "args":["0xTo...", "1000000000000000000"], "gasLimit":null}
    """
    data = request.get_json(force=True)
    from_addr = Web3.to_checksum_address(data["from"])
    method = data["method"]
    args = data.get("args", [])
    func = getattr(contract.functions, method)(*args)

    # Estimate gas & build
    try:
        gas = data.get("gasLimit") or func.estimate_gas({"from": from_addr})
        tx = func.build_transaction({
            "from": from_addr,
            "nonce": w3.eth.get_transaction_count(from_addr),
            "gas": int(gas * 1.2),  # buffer
            "chainId": CHAIN_ID
        })
        return jsonify(tx)
    except Exception as e:
        return abort(400, str(e))

if __name__ == "__main__":
    app.run(debug=True)
