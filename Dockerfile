# ===== EvalSystem Dockerfile =====
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for WeasyPrint (optional PDF support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    libffi-dev libgdk-pixbuf2.0-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy application
COPY . .

# Create output directory
RUN mkdir -p output

# Default port
EXPOSE 8765

# Run dashboard
CMD ["python", "-m", "src.web_dashboard", "--host", "0.0.0.0", "--port", "8765"]
