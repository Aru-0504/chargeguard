FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./app/
COPY agent/ ./agent/

# Ensure model and metrics files exist (create placeholders if not present)
RUN if [ ! -f app/chargeback_model.pkl ]; then \
    echo "Warning: Model file not found. Please ensure chargeback_model.pkl is present."; \
    fi
RUN if [ ! -f app/metrics.json ]; then \
    echo '{"features":["Amount","tx_count_24h","minutes_since_last_tx","amount_vs_card_avg","is_odd_hour"],"threshold":0.8553,"auc":0.9035,"precision":0.7053,"recall":0.5877}' > app/metrics.json; \
    fi

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check (supports dynamic PORT with fallback to 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c "curl -f http://localhost:\${PORT:-8000}/health || exit 1"

# Run the application (supports Render / Railway dynamic $PORT with fallback to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
