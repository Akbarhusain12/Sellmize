from datetime import timedelta
import os
from dotenv import load_dotenv

# 1. Load variables from .env file immediately
load_dotenv()

class Config:
    """Configuration class for Flask app."""
    
    SECRET_KEY = os.environ.get('FLASK_SECRET') or 'a-super-secret-key'
    
    # Folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'output')
    SESSION_FILE_DIR = os.path.join(BASE_DIR, 'flask_session_data')



    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    
    SESSION_TYPE = "filesystem"
    SESSION_PERMANENT = False
    
    # API Keys
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')

    # 2. Database Setup
    # We fetch the URL. If it's missing, we want to know (don't fallback to sqlite silently)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

    if not SQLALCHEMY_DATABASE_URI:
        raise ValueError("DATABASE_URL is not set in environment variables.")


    # Fix for Render/Heroku postgres URLs (they start with postgres:// but SQLAlchemy wants postgresql://)
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False