# notify.py
import os
from flask_socketio import SocketIO

_socketio = SocketIO(message_queue=os.getenv("SOCKETIO_REDIS_URL", "redis://127.0.0.1:6379/2"))

def emit_to_room(event: str, room: str, data: dict):
    # room examples: f"user:{user_id}" or f"job:{job_id}"
    _socketio.emit(event, data, to=room)

def broadcast(event: str, data: dict):
    _socketio.emit(event, data, broadcast=True)
