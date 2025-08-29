# celery_utils.py
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        backend=app.config["CELERY_RESULT_BACKEND"],
        include=["tasks.gpu_tasks"]
    )
    celery.conf.update(
        task_track_started=app.config.get("CELERY_TASK_TRACK_STARTED", True),
        task_time_limit=app.config.get("CELERY_TASK_TIME_LIMIT", 28800),
        broker_connection_retry_on_startup=True,
        worker_send_task_events=True,
        task_send_sent_event=True,
        result_expires=86400,
        acks_late=app.config.get("CELERY_ACKS_LATE", True),
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
