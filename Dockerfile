FROM python:3.10-slim

# Opsional: percepat build
ENV PIP_NO_CACHE_DIR=1  PYTHONDONTWRITEBYTECODE=1  PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# HF Spaces akan set env PORT; default 7860
ENV PORT=7860

EXPOSE 7860

# Gunicorn untuk production
CMD ["gunicorn", "-w", "2", "-k", "gevent", "-b", "0.0.0.0:${PORT}", "wsgi:application"]
