import numpy as np

class User:
    """ Represents a user with a wallet to receive and send payments. """
    def __init__(self, username, initial_balance=0):
        self.username = username
        self.wallet = initial_balance

    def deposit(self, amount):
        """ Add funds to the user's wallet. """
        self.wallet += amount

    def withdraw(self, amount):
        """ Withdraw funds from the user's wallet (if sufficient balance). """
        if self.wallet >= amount:
            self.wallet -= amount
            return True
        else:
            print(f"Insufficient funds for {self.username}. Transaction failed.")
            return False

    def __str__(self):
        return f"{self.username}: Wallet Balance = ${self.wallet:.2f}"

class SwapContract:
    """ Represents an equity swap contract with approval-based cash flow transfers. """
    def __init__(self, notional, fixed_rate, seller: User, buyer: User):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.seller = seller
        self.buyer = buyer
        self.pending_transfers = []  # Store transactions requiring approval

    def request_transfer(self, periods):
        """
        Request payments for a swap contract.
        
        - Fixed payments come from buyer → seller
        - Floating payments come from seller → buyer
        - Net cash flow determines final direction of payment.

        Returns:
        - A list of pending transfers.
        """
        pending_cash_flows = []
        for t in range(1, periods + 1):
            fixed_payment = self.notional * self.fixed_rate  # Fixed leg payment
            floating_payment = fixed_payment * (1 + np.random.normal(0, 0.01))  # Floating leg
            net_cash_flow = floating_payment - fixed_payment  # Net difference

            # Determine payment direction
            if net_cash_flow > 0:
                payer, receiver = self.seller, self.buyer
            else:
                payer, receiver = self.buyer, self.seller

            transfer = {
                "period": t,
                "fixed_payment": fixed_payment,
                "floating_payment": floating_payment,
                "net_cash_flow": abs(net_cash_flow),
                "payer": payer.username,
                "receiver": receiver.username,
                "status": "Pending"
            }
            pending_cash_flows.append(transfer)

        self.pending_transfers.extend(pending_cash_flows)
        return pending_cash_flows

    def approve_transfer(self):
        """
        Approve all pending transfers and process payments between users.
        """
        executed_cash_flows = []
        for transfer in self.pending_transfers:
            payer = self.seller if transfer["payer"] == self.seller.username else self.buyer
            receiver = self.buyer if transfer["receiver"] == self.buyer.username else self.seller

            # Attempt transfer
            if payer.withdraw(transfer["net_cash_flow"]):
                receiver.deposit(transfer["net_cash_flow"])
                transfer["status"] = "Approved"
                executed_cash_flows.append(transfer)

        # Clear processed transactions
        self.pending_transfers = []
        return executed_cash_flows

# Example Usage
seller = User("Seller_ABC", initial_balance=50000)
buyer = User("Buyer_XYZ", initial_balance=50000)

swap = SwapContract(notional=1_000_000, fixed_rate=0.03, seller=seller, buyer=buyer)



## Step 1: Request Swap Transfer (Requires Approval)
#pending_transfers = swap.request_transfer(periods=5)
#print("Pending Transfers (Require Approval):")
#for transfer in pending_transfers:
#   print(transfer)
#
## Step 2: Approve and Execute Transfers
#approved_transfers = swap.approve_transfer()
#print("\nApproved Transfers (Executed):")
#for transfer in approved_transfers:
#   print(transfer)
#
## Step 3: Print Wallet Balances After Transactions
#print("\nFinal Wallet Balances:")
#print(seller)
#print(buyer)
