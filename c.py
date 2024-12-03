from cdp import *

# Paste in your API key name and private key generated from https://portal.cdp.coinbase.com/access/api below:
api_key_name = "organizations/your-org-id/apiKeys/your-api-key-id";

api_key_private_key = "-----BEGIN EC PRIVATE KEY-----\nyour-api-key-private-key\n-----END";

Cdp.configure(api_key_name, api_key_private_key)

# Create your first wallet.
wallet = Wallet.create()

# Fund your wallet with a faucet.
faucet_tx = wallet.faucet()