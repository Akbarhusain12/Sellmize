# Gunicorn configuration file
import multiprocessing
import os

# Bind to the PORT environment variable (Render requirement)
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker configuration
workers = 1  # Use only 1 worker on free tier to save memory
worker_class = 'sync'
threads = 1

# Timeout settings - CRITICAL for ML processing
timeout = 300  # 5 minutes (increased from 30 seconds)
graceful_timeout = 300
keepalive = 5

# Memory and performance
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 20
preload_app = False  # Don't preload to save memory

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log errors to stdout
loglevel = 'info'

# Limits
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190