from flask import Flask, render_template, request, redirect, abort, jsonify,sessions, Response, url_for,send_file,render_template_string,flash
from flask import Blueprint
import os
import csv

app = Flask(__name__)

node = None

@app.route('/exchange', methods=['GET','POST'])
def exchange():
	return render_template("p2p.html")

@app.route('/start', methods=['POST'])
def start_node():
	global node
	data = request.json
	host = data.get('host', '127.0.0.1')
	port = data.get('port', 4040)
	if node is None:
		node = P2PNode(host, port)
		node.start()
		return jsonify({'message': f'Started node on {host}:{port}'})
	return jsonify({'message': 'Node already running'})

@app.route('/connect', methods=['POST'])
def connect_peer():
	global node
	if node is None:
		return jsonify({'error': 'Node not started'}), 400
	
	data = request.json
	peer_host = data.get('peer_host')
	peer_port = data.get('peer_port')
	
	if not peer_host or not peer_port:
		return jsonify({'error': 'peer_host and peer_port are required parameters'}), 400
	
	try:
		peer_port = int(peer_port)
	except ValueError:
		return jsonify({'error': 'peer_port must be a valid integer'}), 400
	
	node.connect_to_peer(peer_host, peer_port)
	return jsonify({'message': f'Connected to peer {peer_host}:{peer_port}'})

@app.route('/send', methods=['POST'])
def send_message():
	global node
	if node is None:
		return jsonify({'error': 'Node not started'}), 400
	
	data = request.json
	message = data.get('message')
	if not message:
		return jsonify({'error': 'Message is required'}), 400
	
	node.send_message(message)
	return jsonify({'message': f'Message sent: {message}'})

app.run()