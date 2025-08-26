# extensions.py
import os
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

# Optional Redis (auto-detect). If not present, the state manager will fall back to memory.
redis_client = None
try:
    import redis
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
    if url:
        if url.startswith("redis://"):
            redis_client = redis.Redis.from_url(url)
        else:
            redis_client = redis.Redis(host=url, port=int(os.getenv("REDIS_PORT", "6379")), decode_responses=False)
except Exception:
    redis_client = None
