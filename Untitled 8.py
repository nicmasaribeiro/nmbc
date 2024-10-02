#!/usr/bin/env python3

import asyncio

async def handle_client(reader, writer):
	peername = writer.get_extra_info('peername')
	print(f"Connection from {peername}")
	
	while True:
		data = await reader.read(100)
		message = data.decode()
		
		if not data:
			print("Connection closed by the client")
			break
		
		print(f"Received: {message}")
		response = "Message received"
		writer.write(response.encode())
		await writer.drain()
		
	writer.close()
	
async def main():
	server = await asyncio.start_server(handle_client, '127.0.0.1', 8888)
	address = server.sockets[0].getsockname()
	print(f'Serving on {address}')
	
	async with server:
		await server.serve_forever()
		
asyncio.run(main())