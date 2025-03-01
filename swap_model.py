from flask import Blueprint, render_template, request, redirect, url_for
from models import *
import hashlib
import numpy as np
import json
from datetime import datetime

class TokenizedInterestRateSwap:
    def __init__(self, swap_id, issuer, notional, fixed_rate, floating_rate_spread, counterparty_a, counterparty_b):
        self.swap_id = swap_id
        self.issuer = issuer
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate_spread = floating_rate_spread
        self.counterparty_a = counterparty_a
        self.counterparty_b = counterparty_b
        self.name = f"TokenizedSwap-{swap_id}"
        self.symbol = f"TS{swap_id}"
        self.pending_transfers = []  # Store transactions requiring approval

    def calculate_hash(self, index, previous_hash, data):
        """Compute SHA256 hash for a block."""
        return hashlib.sha256(f"{index}{previous_hash}{data}".encode()).hexdigest()

    def create_genesis_block(self):
        """Create the first block in the swap's blockchain."""
        genesis_data = json.dumps({'message': 'Genesis Block'})
        
        genesis_block = SwapBlock(
            index=0,
            timestamp=datetime.utcnow(),
            data=genesis_data,
            previous_hash='0',
            hash=self.calculate_hash(0, '0', genesis_data),
            swap_id=self.swap_id,
            status="Pending"  # ✅ Assign a default status
        )
        
        db.session.add(genesis_block)
        db.session.commit()


    def add_transaction(self, period, floating_rate_index, fixed_payment, floating_payment, net_cash_flow):
        """Add a new transaction block to the blockchain."""
        transaction_data = json.dumps({
            'period': period,
            'floating_rate_index': floating_rate_index,
            'fixed_payment': fixed_payment,
            'floating_payment': floating_payment,
            'net_cash_flow': net_cash_flow,
            'counterparty_a': self.counterparty_a,
            'counterparty_b': self.counterparty_b
        })

        previous_block = SwapBlock.query.filter_by(swap_id=self.swap_id).order_by(SwapBlock.index.desc()).first()
        
        if not previous_block:
            self.create_genesis_block()
            previous_block = SwapBlock.query.filter_by(swap_id=self.swap_id).order_by(SwapBlock.index.desc()).first()

        new_block = SwapBlock(
            index=previous_block.index + 1,
            timestamp=datetime.utcnow(),
            data=transaction_data,
            previous_hash=previous_block.hash,
            hash=self.calculate_hash(previous_block.index + 1, previous_block.hash, transaction_data),
            swap_id=self.swap_id
        )

        db.session.add(new_block)
        db.session.commit()

    def request_transfer(self, periods):
        """Request a transfer by generating pending transactions."""
        fixed_payment = self.notional * self.fixed_rate
        floating_payment = fixed_payment * (1 + np.random.normal(0, 0.01, periods))  # Vectorized computation
        net_cash_flows = floating_payment - fixed_payment

        self.pending_transfers.extend([
            {
                "period": t + 1,
                "fixed_payment": fixed_payment,
                "floating_payment": floating_payment[t],
                "net_cash_flow": net_cash_flows[t],
                "status": "Pending"
            }
            for t in range(periods)
        ])
        return self.pending_transfers  # Return list of pending transfers for review

    def approve_transfer(self):
        """Approve and execute all pending transfers."""
        if not self.pending_transfers:
            return []

        # Collect unique payer and receiver addresses
        payer_addresses = {t["payer"] for t in self.pending_transfers if "payer" in t}
        receiver_addresses = {t["receiver"] for t in self.pending_transfers if "receiver" in t}

        # Fetch wallets in a single query
        payer_wallets = {w.address: w for w in WalletDB.query.filter(WalletDB.address.in_(payer_addresses)).all()}
        receiver_wallets = {w.address: w for w in WalletDB.query.filter(WalletDB.address.in_(receiver_addresses)).all()}

        executed_cash_flows = []
        new_transactions = []

        for transfer in self.pending_transfers:
            payer_wallet = payer_wallets.get(transfer.get("payer"))
            receiver_wallet = receiver_wallets.get(transfer.get("receiver"))

            if payer_wallet and receiver_wallet and payer_wallet.balance >= transfer["net_cash_flow"]:
                payer_wallet.balance -= transfer["net_cash_flow"]
                receiver_wallet.balance += transfer["net_cash_flow"]
                transfer["status"] = "Approved"
                executed_cash_flows.append(transfer)

                new_transactions.append(SwapTransaction(
                    swap_id=self.swap_id,
                    sender=transfer["payer"],
                    receiver=transfer["receiver"],
                    amount=transfer["net_cash_flow"],
                    status="Approved",
                    timestamp=datetime.utcnow()
                ))

        # Batch insert new transactions
        db.session.add_all(new_transactions)
        db.session.commit()

        self.pending_transfers.clear()  # Clear only after processing all
        return executed_cash_flows

    def simulate_payments(self, floating_rate_indices):
        """Simulate payments over multiple periods based on floating rate indices."""
        for period, floating_rate_index in enumerate(floating_rate_indices, start=1):
            fixed_payment = self.notional * self.fixed_rate
            floating_payment = self.notional * (floating_rate_index + self.floating_rate_spread)
            net_cash_flow = floating_payment - fixed_payment
            self.add_transaction(period, floating_rate_index, fixed_payment, floating_payment, net_cash_flow)
