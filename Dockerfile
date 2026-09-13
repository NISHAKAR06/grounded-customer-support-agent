FROM python:3.11-slim

WORKDIR /app

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production \
    HF_HOME=/app/models/cache

# Install minimal OS utilities (including libgomp1 required by faiss-cpu OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (with lightweight CPU-only torch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, data splits, models, templates, and scripts
COPY app/ app/
COPY data/ data/
COPY models/ models/
COPY templates/ templates/
COPY static/ static/
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY docs/ docs/
COPY pyproject.toml .

# Build baseline classifiers and FAISS vector index inside container
RUN python scripts/training/train_baselines.py && \
    python scripts/training/build_faiss_index.py

# Expose web server port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
