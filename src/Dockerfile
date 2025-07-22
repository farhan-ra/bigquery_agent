FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED 1
WORKDIR /api
COPY . /api
COPY requirements.txt /api/
COPY .env /api/


# Install wget
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["python", "bigquery.py"]
