FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (tesseract-ocr for image OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code -- copy all source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]