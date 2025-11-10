import os

class Config:
    """Configuration class for Flask app."""
    
    # Secret key is needed for flashing messages
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-super-secret-key-you-should-change'
    
    # Directory to temporarily store user uploads
    UPLOAD_FOLDER = 'static/uploads'
    
    # Directory to temporarily store generated reports
    OUTPUT_FOLDER = 'static/output'

    # Optional: Set a max upload size (e.g., 100MB)
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
