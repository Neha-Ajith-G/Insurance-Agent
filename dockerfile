FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from the app code, so Docker's
# layer cache only reinstalls packages when requirements.txt actually
# changes -- not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .
COPY docloader.py .
COPY docclient.py .
COPY docmodels.py .
COPY docparsers.py .
COPY docorchestrator.py .
COPY doc_pipeline_logger.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]