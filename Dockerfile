# Dockerfile
FROM python:3.10-slim


# OS deps (selon TensorFlow/ES)
RUN apt-get update && apt-get install -y \
build-essential git curl && rm -rf /var/lib/apt/lists/*


# Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"


WORKDIR /app


# Copier pyproject + lock d'abord (cache layers)
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
&& poetry install --no-interaction --no-ansi --only main


# Copier le code
COPY . .


# Exposer le port
EXPOSE 8080


# Variables (optionnel — sinon passer via `docker run -e`)
ENV LOOK_BACK=2016 \
INTERVAL_MINUTES=5


# Lancer l’API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]