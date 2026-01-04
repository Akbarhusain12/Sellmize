import os
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing in environment variables.")
