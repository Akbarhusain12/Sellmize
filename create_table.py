from app import create_app
from app.db.connection import database as db

app = create_app()

with app.app_context():
    db.create_all()
    print("✅ Tables created successfully")
