import hashlib
import numpy as np
from models import *

class TokenizedInterestRateSwap:
    def __init__(self, swap_id, issuer, notional, fixed_rate, floating_rate_spread, counterparty_a, counterparty_b):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate_spread = floating_rate_spread
        self.counterparty_a = counterparty_a
        self.counterparty_b = counterparty_b
        self.name = f"TokenizedSwap-{swap_id}"
        self.symbol = f"TS{swap_id}"
        self.issuer = issuer
        self.pending_transfers = []  # Store transactions requiring approval
        self.approvals = {}  # Track approvals (Key: transaction ID, Value: Set of approved counterparties)

    def request_transfer(self, periods):
        pending_cash_flows = []
        for t in range(1, periods + 1):
            fixed_payment = self.notional * self.fixed_rate
            floating_payment = fixed_payment * (1 + np.random.normal(0, 0.01))
            net_cash_flow = floating_payment - fixed_payment

            transaction_id = hashlib.sha256(f"{t}{self.name}".encode()).hexdigest()

            pending_cash_flows.append({
                "transaction_id": transaction_id,
                "period": t,
                "fixed_payment": fixed_payment,
                "floating_payment": floating_payment,
                "net_cash_flow": net_cash_flow,
                "status": "Pending",
                "approvals": set()  # Track which parties have approved
            })
            
            self.approvals[transaction_id] = set()

        self.pending_transfers.extend(pending_cash_flows)
        return pending_cash_flows  # Return list of pending transfers for review

    def approve_transfer(self, transaction_id, approver):
        for transfer in self.pending_transfers:
            if transfer["transaction_id"] == transaction_id:
                # Only counterparties can approve
                if approver in [self.counterparty_a, self.counterparty_b]:
                    self.approvals[transaction_id].add(approver)

                    # Check if both counterparties approved
                    if self.counterparty_a in self.approvals[transaction_id] and self.counterparty_b in self.approvals[transaction_id]:
                        return self.execute_transfer(transfer)

        return {"status": "Waiting for second approval"}

    def execute_transfer(self, transfer):
        payer_wallet = WalletDB.query.filter_by(address=self.counterparty_a).first()
        receiver_wallet = WalletDB.query.filter_by(address=self.counterparty_b).first()

        if payer_wallet and receiver_wallet:
            if payer_wallet.balance >= transfer["net_cash_flow"]:
                payer_wallet.balance -= transfer["net_cash_flow"]
                receiver_wallet.balance += transfer["net_cash_flow"]
                transfer["status"] = "Approved"

                new_transaction = SwapTransaction(
                    swap_id=transfer["transaction_id"],
                    sender=self.counterparty_a,
                    receiver=self.counterparty_b,
                    amount=transfer["net_cash_flow"],
                    status="Approved",
                    timestamp=datetime.utcnow()
                )
                db.session.add(new_transaction)
                db.session.commit()

                # Remove from pending transfers after execution
                self.pending_transfers = [t for t in self.pending_transfers if t["transaction_id"] != transfer["transaction_id"]]
                return {"status": "Transfer executed"}
        
        return {"status": "Insufficient funds or wallet issue"}
