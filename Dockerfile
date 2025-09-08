FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to INFO if not provided
ENV LOG_LEVEL=INFO

# Entrypoint starts workers via windmill.sh; overridable at runtime
ENTRYPOINT ["/bin/bash", "-lc", "./windmill.sh start && tail -F logs/*.log || tail -f /dev/null"]


