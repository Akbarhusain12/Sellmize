import os
from flask import Flask
from flask_login import LoginManager
from .config import Config
from app.db.connection import database as db
from app.db.connection import init_app as init_db

from app.api.routes_dashboard import dashboard_bp
from app.api.routes_analyzer import analyzer_page, analyzer_api
from app.api.routes_marketlens import marketlens_bp
from app.api.routes_content import content_bp
from app.api.routes_auth import auth_bp
from app.db.models import User

def create_app():
    app = Flask(__name__)

    # config
    app.config.from_object(Config)

    # database
    init_db(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login' # Where to send non-logged-in users
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        # Queries the DB for the user with this ID
        return User.query.get(int(user_id))

    # blueprints
    app.register_blueprint(dashboard_bp, url_prefix="/")
    app.register_blueprint(analyzer_page, url_prefix="/analyzer")
    app.register_blueprint(analyzer_api)
    app.register_blueprint(auth_bp, url_prefix='/auth')

    app.register_blueprint(marketlens_bp, url_prefix="/marketlens")
    app.register_blueprint(content_bp, url_prefix="/content")


    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
    # print("UPLOAD FOLDER:", app.config["UPLOAD_FOLDER"])
    # print("OUTPUT FOLDER:", app.config["OUTPUT_FOLDER"])
    # print("SESSION DIR:", app.config["SESSION_FILE_DIR"])

    
    # for rule in app.url_map.iter_rules():
    #     print(rule)


    return app
