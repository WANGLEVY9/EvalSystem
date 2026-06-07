# ===== EvalSystem Dockerfile =====
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for WeasyPrint (optional PDF support)
# Note: Debian trixie renamed libgdk-pixbuf2.0-0 -> libgdk-pixbuf-2.0-0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy application
COPY . .

# Create writable directories (output for reports, uploads for user-uploaded instructions)
RUN mkdir -p output && chmod 777 output

# Default port
EXPOSE 8765

# Run dashboard
CMD python -m src.web_dashboard --host 0.0.0.0 --port ${PORT:-8765}
