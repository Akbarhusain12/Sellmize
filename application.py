from app import create_app

# Gunicorn entrypoint expects 'app'
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
