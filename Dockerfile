# Basis-Image
FROM python:3.11-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# UV installieren
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Projektdateien kopieren
COPY requirements.txt requirements-dev.txt ./
COPY ml4t_project/ ./ml4t_project/
COPY setup.py README.md ./

# Virtuelle Umgebung erstellen und Abhängigkeiten installieren
RUN uv venv .venv \
    && . .venv/bin/activate \
    && uv pip install -r requirements.txt \
    && uv pip install -r requirements-dev.txt

# Port für Monitoring-Dashboard
EXPOSE 8050

# Umgebungsvariablen
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Startbefehl
CMD ["python", "-m", "ml4t_project.main"] 