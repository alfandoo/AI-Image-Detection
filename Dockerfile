FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# HF Spaces akan set PORT; kasih default kalau belum ada
ENV PORT=7860
EXPOSE 7860

# ✅ Pakai shell form agar $PORT diexpand
CMD gunicorn -w 2 -k gevent -b 0.0.0.0:$PORT wsgi:application
