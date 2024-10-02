import socket
import threading

class P2PNode:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []

    def start_server(self):
        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

    def run_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Server started at {self.host}:{self.port}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

    def handle_client(self, client_socket):
        while True:
            try:
                message = client_socket.recv(1024).decode()
                if message:
                    print(f"Received: {message}")
                    self.broadcast(message)
                else:
                    break
            except:
                break
        client_socket.close()

    def connect_to_peer(self, peer_host, peer_port):
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.connect((peer_host, peer_port))
            self.peers.append(peer_socket)
            listen_thread = threading.Thread(target=self.listen_to_peer, args=(peer_socket,))
            listen_thread.start()
        except ConnectionRefusedError:
            print(f"Could not connect to peer at {peer_host}:{peer_port}. Connection refused.")

    def listen_to_peer(self, peer_socket):
        while True:
            try:
                message = peer_socket.recv(1024).decode()
                if message:
                    print(f"Received from peer: {message}")
                else:
                    break
            except:
                break
        peer_socket.close()

    def broadcast(self, message):
        for peer_socket in self.peers:
            try:
                peer_socket.sendall(message.encode())
            except:
                self.peers.remove(peer_socket)

    def send_message(self, message):
        self.broadcast(message)

# if __name__ == "__main__":
    # # Starting the node server
    # host = "127.0.0.1"  # Use '127.0.0.1' for localhost testing or a specific IP address
    # port = 12345
    # node = P2PNode(host, port)
    
    # # Start the server thread
    # node.start_server()

    # # Connect to a peer
    # # Note: You should have another peer running on the specified host and port
    # peer_host = "127.0.0.1"
    # peer_port = 12345  # Ensure that a peer is actually running at this address and port
    # node.connect_to_peer(peer_host, peer_port)

    # # Sending messages
#   while True:
#       message = input("Enter message to broadcast: ")
#       node.send_message(message)