
from pywallet import wallet

# Generate a new mnemonic (12-word seed phrase)
mnemonic = wallet.generate_mnemonic()

# Create a wallet from the mnemonic
w = wallet.create_wallet(network="BTC", seed=mnemonic, children=1)

# Get the master key
master_key = w.get("master")

# Get the first Bitcoin address
address = w.get("address")

# Get the public and private keys
public_key = w.get("public_key")
private_key = w.get("private_key")


# Print your wallet details
print("Mnemonic:", mnemonic)
print("Master Key:", master_key)
print("Address:", address)
print("Public Key:", public_key)
print("Private Key:", private_key)