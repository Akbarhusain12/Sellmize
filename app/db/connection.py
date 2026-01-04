import os
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

database = SQLAlchemy()


def init_app(app):
    db_url = os.getenv("DATABASE_URL")


    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment variables")

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    database.init_app(app)
