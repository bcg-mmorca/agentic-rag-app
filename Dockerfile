FROM python:3.13-slim

WORKDIR /app

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python -m venv /app/venv

# Activate the virtual environment and install dependencies
COPY requirements.txt .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the application port
EXPOSE 4000

# Define volumes for persistent data
VOLUME [ "/chroma_langchain_db", "/secrets" ]

# Start the application using the virtual environment
CMD ["/app/venv/bin/uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "4000"]
