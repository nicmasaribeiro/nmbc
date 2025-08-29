# models_async.py (or inside your existing models.py)
from datetime import datetime
from app import db  # adjust import if your db is elsewhere

class AsyncJob(db.Model):
    __tablename__ = "async_jobs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True, nullable=True)

    task_name = db.Column(db.String(120), nullable=False)
    celery_id = db.Column(db.String(64), index=True)

    status = db.Column(db.String(32), default="queued")   # queued, running, succeeded, failed
    progress = db.Column(db.Float, default=0.0)

    prefer_gpu = db.Column(db.Boolean, default=True)
    gpu_id = db.Column(db.Integer, nullable=True)
    queue = db.Column(db.String(32), default="cpu")

    payload_json = db.Column(db.Text, nullable=True)
    result_json = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)

    notify_channel = db.Column(db.String(128), nullable=True)  # e.g. f"user:{user_id}" or f"job:{id}"

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
