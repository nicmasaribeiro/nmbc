# config.py (or inside app.config[...] anywhere you configure Flask)
CELERY_BROKER_URL = "redis://127.0.0.1:6379/0"
CELERY_RESULT_BACKEND = "redis://127.0.0.1:6379/1"
SOCKETIO_REDIS_URL = "redis://127.0.0.1:6379/2"

# optional celery tuning
CELERY_TASK_TIME_LIMIT = 60 * 60 * 8   # 8 hours
CELERY_TASK_SOFT_TIME_LIMIT = 60 * 60 * 8
CELERY_ACKS_LATE = True
CELERY_TASK_TRACK_STARTED = True
