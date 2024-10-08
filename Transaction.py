#!/usr/bin/env python3

import binascii


class Transaction:
	def __init__(self, sender: str, recipient: str, amount: float, signature: str = None):
		self.sender = sender
		self.recipient = recipient
		self.amount = amount
		self.signature = signature
		
	def to_dict(self):
		return {
			'sender': self.sender,
			'recipient': self.recipient,
			'amount': self.amount
		}
	
	def sign_transaction(self, private_key: str):
		private_key = RSA.import_key(binascii.unhexlify(private_key))
		signer = PKCS1_v1_5.new(private_key)
		h = SHA256.new(str(self.to_dict()).encode('utf8'))
		self.signature = binascii.hexlify(signer.sign(h)).decode('ascii')
		
	def is_valid(self):
		if self.sender == "MINING":
			return True
		public_key = RSA.import_key(binascii.unhexlify(self.sender))
		verifier = PKCS1_v1_5.new(public_key)
		h = SHA256.new(str(self.to_dict()).encode('utf8'))
		return verifier.verify(h, binascii.unhexlify(self.signature))
	