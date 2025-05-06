FROM python:3.10-slim

# System deps for audio handling
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
