#!/usr/bin/env python3

import redis

try:
	r = redis.Redis.from_url("rediss://your-redis-host:6379")
	r.ping()
	print("Connected successfully!")
except redis.exceptions.ConnectionError as e:
	print(f"Connection failed: {e}")
	