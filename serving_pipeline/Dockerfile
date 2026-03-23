FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY load_model.py .
COPY preprocessing.py .
COPY monitoring.py .
COPY api/ ./api/
RUN mkdir -p ./api/data ./api/reports
EXPOSE 8000
WORKDIR /app/api
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
