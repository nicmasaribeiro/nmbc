import socket
import threading

class P2PNode:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []  # List of connected peers

    def start(self):
        # Start server thread to listen for incoming connections
        server_thread = threading.Thread(target=self.listen_for_peers)
        server_thread.start()

    def listen_for_peers(self):
        # Create a socket for listening for incoming peer connections
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Listening for connections on {self.host}:{self.port}...")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connected to {client_address}")
            self.peers.append(client_socket)

            # Start a new thread to handle the connection
            threading.Thread(target=self.handle_peer_connection, args=(client_socket,)).start()

    def handle_peer_connection(self, client_socket):
        try:
            while True:
                # Receive message from the peer
                message = client_socket.recv(1024).decode('utf-8')
                if not message:
                    break
                print(f"Received: {message}")

                # Broadcast the message to all other peers
                self.broadcast_message(message, client_socket)
        except ConnectionResetError:
            print("Connection closed by peer.")
        finally:
            client_socket.close()
            self.peers.remove(client_socket)

    def broadcast_message(self, message, exclude_socket=None):
        # Broadcast a message to all connected peers, except the sender
        for peer in self.peers:
            if peer != exclude_socket:
                try:
                    peer.sendall(message.encode('utf-8'))
                except BrokenPipeError:
                    print("Failed to send message. Removing peer.")
                    self.peers.remove(peer)

    def connect_to_peer(self, peer_host, peer_port):
        # Connect to a known peer
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.connect((peer_host, peer_port))
            self.peers.append(peer_socket)

            # Start a thread to handle incoming messages from this peer
            threading.Thread(target=self.handle_peer_connection, args=(peer_socket,)).start()

            print(f"Connected to peer {peer_host}:{peer_port}")

            # Send a welcome message (optional)
            peer_socket.sendall("Hello from new peer!".encode('utf-8'))
        except (socket.error, ConnectionRefusedError) as e:
            print(f"Failed to connect to {peer_host}:{peer_port}: {e}")

    def send_message(self, message):
        # Send a message to all connected peers
        print(f"Sending: {message}")
        self.broadcast_message(message)


#if __name__ == "__main__":
#   # Example usage of P2PNode
#   host = '127.0.0.1'  # Localhost
#   port = 4040         # Port to listen on
#
#   node = P2PNode(host, port)
#   node.start()
#
#   # Connect to another peer (optional)
#   # node.connect_to_peer('127.0.0.1', 8081)
#
#   # Send messages manually (for testing purposes)
#   while True:
#       msg = input("Enter")
