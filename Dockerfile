FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y nginx

RUN pip install Flask gunicorn transformers torch Pillow

COPY ./app.py /app/
COPY ./nginx.conf /etc/nginx/sites-available/default

CMD service nginx start && gunicorn app:app -b 0.0.0.0:8000
