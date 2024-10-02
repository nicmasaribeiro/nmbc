import websocket

# Ignore SSL verification errors
ws = websocket.WebSocket(sslopt={"cert_reqs": 0})
ws.connect("wss://go.getblock.io/ebd51366cb214aa38e6477d2f30e167c")

# Sending a sample message
ws.send("Hello, server!")
response = ws.recv()
print(response)

ws.close()