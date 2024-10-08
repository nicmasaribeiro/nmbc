from cdp import *

api_key = "organizations/5eb0bfbc-3029-4b75-aac6-39ba188d3ac5/apiKeys/ee424a62-beb9-4673-9ef6-7abf2af0d612"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIIrFOL9aVS7DRHGkY8/vyuDIDdW8JBeNf6oraa5c7riOoAoGCCqGSM49\nAwEHoUQDQgAEL5Vod5wi+tHXRmn7aiwwnd12d8brinhlrQsk1nJmeQEC8JpFqAJ+\nTmPiJ3r00ZG3UFuJbGsip9Yia1F+4nAiEQ==\n-----END EC PRIVATE KEY-----\n"
Cdp.configure(api_key, api_secret)

# Create a wallet (or import an existing one)
wallet = Wallet.create()
address = wallet.create_address()
print(address.address_id)

print("\nBalance\n",wallet.balance(address))



# Since this is an ERC-721 NFT, there's no need to define the ABI


# Fund the wallet manually or via `wallet.faucet`

# Deploy the token
# deployed_contract = wallet.deploy_token('nmbcoin', "api", 100)
# deployed_contract.wait()

# # Interact with the deployed contract
# invocation = wallet.invoke_contract(
#   contract_address=deployed_contract.contract_address, 
#   method="approve", 
#   args={"spender": "0xApprovedSpender", "value": "10000000"})
# invocation.wait()

# import requests
# import json

# url = 'https://mainnet.infura.io/v3/38259072056b41d88b5ecc0c23cc02fa'

# payload = {
#     "jsonrpc": "2.0",
#     "method": "eth_blockNumber",
#     "params": [],
#     "id": 1
# }

# headers = {'content-type': 'application/json'}

# response = requests.post(url, data=json.dumps(payload), headers=headers).json()

# print(response)



# from web3 import Web3

# Connect to Ethereum (replace with your provider)
# infura_url = "https://mainnet.infura.io/v3/38259072056b41d88b5ecc0c23cc02fa"
# web3 = Web3(Web3.HTTPProvider(infura_url))

# # ABI of your contract (copy this from the contract deployment)
# contract_abi = '''[YOUR_CONTRACT_ABI]'''

# # Contract address (replace with your contract address)
# contract_address = "0xYourContractAddress"

# # Load contract
# my_token_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# # Interact with contract
# @app.route('/mint_tokens/<user_address>/<amount>', methods=['POST'])
# def mint_tokens(user_address, amount):
#     # Set up the account that will send the transaction (contract owner)
#     private_key = "YOUR_PRIVATE_KEY"
#     owner_address = "0xOwnerAddress"

#     # Convert amount to integer
#     amount = int(amount)

#     # Prepare transaction for minting tokens
#     nonce = web3.eth.getTransactionCount(owner_address)
#     txn = my_token_contract.functions.mint(user_address, amount).buildTransaction({
#         'chainId': 1,  # Change to your chain (e.g., 1 for Ethereum Mainnet)
#         'gas': 2000000,
#         'gasPrice': web3.toWei('20', 'gwei'),
#         'nonce': nonce
#     })

#     # Sign transaction
#     signed_txn = web3.eth.account.sign_transaction(txn, private_key)

#     # Send transaction
#     tx_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)

#     # Wait for transaction to be mined
#     tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

#     return f"Mint transaction hash: {tx_receipt.transactionHash.hex()}"