# Etapa base
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PATH="/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv /venv

# Dependencias
FROM base AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Runtime
FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" LOG_LEVEL=INFO PORT=8000
WORKDIR /app
COPY --from=base /venv /venv
COPY --from=deps /venv /venv
COPY . .
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:create_app()"]
