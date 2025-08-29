# routes/bitcoin.py
import os
from flask import Blueprint, request, jsonify
from services.bitcoin_wallet import BitcoinWalletService

bp = Blueprint("bitcoin", __name__, url_prefix="/api/btc")
svc = BitcoinWalletService()

# ---- In your DB, store: user_id, xpub, next_index, etc. (pseudo) ----
FAKE_DB = {}

@bp.post("/wallet")
def create_wallet():
    # Option A: create mnemonic server-side (custody); Option B: user supplies theirs.
    mnemonic = svc.new_mnemonic()
    seed = svc.seed_from_mnemonic(mnemonic)
    xpub = svc.account_xpub(seed, account=0)
    # Store ONLY xpub + next_index in DB for receive flow; keep mnemonic off-box or encrypted
    FAKE_DB["xpub"] = xpub
    FAKE_DB["next_index"] = 0
    return jsonify({"mnemonic": mnemonic, "xpub": xpub})  # return mnemonic once; never again

@bp.post("/address/new")
def new_receive_address():
    xpub = FAKE_DB["xpub"]
    idx = FAKE_DB["next_index"]
    derived = svc.next_receive_address(xpub, next_index=idx)
    FAKE_DB["next_index"] = idx + 1
    # Persist address → user map in DB to attribute deposits
    return jsonify(derived.dict())

@bp.get("/address/<addr>/utxos")
def utxos(addr):
    data = [u.dict() for u in svc.get_utxos(addr)]
    return jsonify(data)

@bp.post("/send")
def send_tx():
    """
    Body: { "from_addresses": [...], "to_address": "bc1...", "amount_sat": 12345, "change_address": "...", "psbt_only": true/false }
    """
    body = request.get_json(force=True)
    to_address = body["to_address"]
    amount_sat = int(body["amount_sat"])
    from_addrs = body["from_addresses"]
    change_address = body["change_address"]

    # Gather UTXOs
    utxos = []
    for a in from_addrs:
        utxos += svc.get_utxos(a)

    fee_rate = svc.get_fee_rate_sat_vb()
    psbt = svc.build_psbt(utxos, to_address, amount_sat, fee_rate, change_address)

    # Return PSBT-like data (sign client-side or server-side depending on your custody policy)
    return jsonify({
        "psbt_skeleton": psbt,
        "fee_rate_sat_vb": fee_rate
    })
