# utils/locks.py
import time
import redis
from contextlib import contextmanager

r = redis.Redis(host="localhost", port=6379, db=2)

@contextmanager
def redis_lock(name: str, ttl=55):
    token = str(time.time())
    ok = r.set(name, token, nx=True, ex=ttl)  # set if not exists with TTL
    if not ok:
        yield False
        return
    try:
        yield True
    finally:
        # best-effort release (optional compare token if you want full safety)
        r.delete(name)
