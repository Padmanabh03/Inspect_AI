# InspectAI Backend Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY patchcore/ ./patchcore/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY config.py .
COPY models/ ./models/

# Expose port (Render assigns this via $PORT env variable)
EXPOSE 8000

# Run the application (Render uses PORT env variable)
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
