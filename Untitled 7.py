#!/usr/bin/env python3

import asyncio

async def send_message():
	reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
	
	while True:
		message = input("Enter message to send: ")
		writer.write(message.encode())
		await writer.drain()
		
		data = await reader.read(100)
		print(f'Received from server: {data.decode()}')
		
		if message.lower() == "quit":
			print("Closing connection")
			writer.close()
			await writer.wait_closed()
			break
		
asyncio.run(send_message())