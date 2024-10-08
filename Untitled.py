#!/usr/bin/env python3

import asyncio
import websockets
import threading
import json
from websocket import create_connection

class Node:
	def __init__(self, host, port):
		self.host = host
		self.port = port
		self.peers = []  # Store the URLs of connected peers
		self.connections = []  # Store active WebSocket connections
		
	async def start_server(self):
		"""Starts the WebSocket server to accept incoming peer connections."""
		server = await websockets.serve(self.handle_connection, self.host, self.port)
		print(f"WebSocket Server started at ws://{self.host}:{self.port}")
		await server.wait_closed()
		
	async def handle_connection(self, websocket, path):
		"""Handles incoming WebSocket connections from peers."""
		print(f"New connection from peer: {websocket.remote_address}")
		self.connections.append(websocket)
		try:
			async for message in websocket:
				print(f"Received message from {websocket.remote_address}: {message}")
				await self.broadcast(message, websocket)
		except websockets.ConnectionClosed:
			print(f"Connection closed: {websocket.remote_address}")
			self.connections.remove(websocket)
			
	async def broadcast(self, message, sender_websocket):
		"""Broadcasts a message to all connected peers, excluding the sender."""
		for peer in self.connections:
			if peer != sender_websocket:
				try:
					await peer.send(message)
				except websockets.ConnectionClosed:
					print("Connection closed, removing peer")
					self.connections.remove(peer)
					
	def connect_to_peer(self, peer_url):
		"""Connects to another peer using a WebSocket connection."""
		def connect():
			try:
				ws = create_connection(peer_url)
				self.peers.append(peer_url)
				self.connections.append(ws)
				print(f"Connected to peer: {peer_url}")
				while True:
					message = ws.recv()
					print(f"Received message from peer {peer_url}: {message}")
			except Exception as e:
				print(f"Failed to connect to peer {peer_url}: {e}")
				
		threading.Thread(target=connect).start()
		
	def send_message_to_peers(self, message):
		"""Sends a message to all connected peers."""
		for ws in self.connections:
			try:
				if isinstance(ws, create_connection):
					ws.send(message)
				elif isinstance(ws, websockets.WebSocketServerProtocol):
					asyncio.run(ws.send(message))
			except Exception as e:
				print(f"Error sending message to a peer: {e}")
				
if __name__ == "__main__":
	# Node setup: specify host and port for this node
	host = '127.0.0.1'
	port = 8765
	
	node = Node(host, port)
	
	# Start WebSocket server in a separate thread to listen for incoming connections
	server_thread = threading.Thread(target=lambda: asyncio.run(node.start_server()))
	server_thread.start()
	
	# Example: Connect to another peer (a separate node running on a different port)
	peer_url = "ws://127.0.0.1:8766"  # Change to another node URL
	node.connect_to_peer(peer_url)
	
	# Sending a message to all connected peers
	while True:
		user_message = input("Enter a message to send to peers: ")
		node.send_message_to_peers(user_message)