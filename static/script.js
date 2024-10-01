function startNode() {
    const host = document.getElementById('host').value;
    const port = document.getElementById('port').value;

    fetch('/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ host: host, port: parseInt(port) })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('startNodeMessage').textContent = data.message;
    })
    .catch(error => console.error('Error:', error));
}

function connectPeer() {
    const peerHost = document.getElementById('peerHost').value;
    const peerPort = document.getElementById('peerPort').value;

    fetch('/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ peer_host: peerHost, peer_port: parseInt(peerPort) })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('connectMessage').textContent = data.message || data.error;
    })
    .catch(error => console.error('Error:', error));
}

function sendMessage() {
    const message = document.getElementById('message').value;

    fetch('/send', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('sendMessageMessage').textContent = data.message || data.error;
    })
    .catch(error => console.error('Error:', error));
}