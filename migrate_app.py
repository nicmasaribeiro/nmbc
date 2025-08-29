# minimal app just for Alembic/Flask-Migrate
from models import app, db  # <- your models.py already creates app = Flask(__name__) and db = SQLAlchemy(app)
from flask_migrate import Migrate

migrate = Migrate(app, db)

# Optional: expose a CLI entry point if you ever run python migrate_app.py
if __name__ == "__main__":
    app.run()
