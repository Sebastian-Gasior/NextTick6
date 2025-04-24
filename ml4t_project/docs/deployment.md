# Deployment Guide

Dieser Guide beschreibt die Schritte zum Deployment des ML4T-Projekts in einer Produktionsumgebung.

## 🐳 Docker Deployment

### Dockerfile erstellen

```dockerfile
FROM python:3.8-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install uv && \
    uv pip install -r requirements.txt

# Projektdateien kopieren
COPY ml4t_project/ .

# Port für API freigeben
EXPOSE 8000

# Startbefehl
CMD ["python", "main.py"]
```

### Container bauen und starten

```bash
# Container bauen
docker build -t ml4t:latest .

# Container starten
docker run -d -p 8000:8000 ml4t:latest
```

## 🚀 CI/CD Pipeline (GitHub Actions)

`.github/workflows/main.yml`:

```yaml
name: ML4T CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install uv
        uv pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest ml4t_project/tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        echo "Deployment steps here"
```

## 📊 Monitoring

### Prometheus Metriken

`monitoring.py`:
```python
from prometheus_client import Counter, Gauge, start_http_server

# Metriken definieren
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made')
MODEL_LATENCY = Gauge('model_latency_seconds', 'Time taken for prediction')
ACCURACY = Gauge('model_accuracy', 'Current model accuracy')

# Metriken-Server starten
start_http_server(8000)
```

### Grafana Dashboard

`grafana/dashboard.json`:
```json
{
  "dashboard": {
    "title": "ML4T Monitoring",
    "panels": [
      {
        "title": "Predictions per Minute",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(predictions_total[5m])",
            "legendFormat": "Predictions"
          }
        ]
      },
      {
        "title": "Model Latency",
        "type": "gauge",
        "targets": [
          {
            "expr": "model_latency_seconds",
            "legendFormat": "Latency"
          }
        ]
      }
    ]
  }
}
```

## 🔄 Updates und Rollbacks

### Model Versioning

```python
from ml4t_project.model.versioning import ModelVersion

# Modell speichern
version = ModelVersion(
    model=trained_model,
    metrics=validation_metrics,
    timestamp=datetime.now()
)
version.save()

# Rollback zu vorheriger Version
previous_version = ModelVersion.load('v1.2.3')
model = previous_version.model
```

## 🔐 Sicherheit

1. **API-Sicherheit**
   - JWT-Authentication
   - Rate Limiting
   - Input Validation

2. **Daten-Sicherheit**
   - Verschlüsselte Speicherung
   - Regelmäßige Backups
   - Zugriffsprotokollierung

3. **Modell-Sicherheit**
   - Versionsmanagement
   - A/B Testing
   - Monitoring von Drift

## 📝 Logging

```python
import logging

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml4t.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ml4t')
```

## 🔍 Health Checks

```python
from healthcheck import HealthCheck

health = HealthCheck()

def model_available():
    """Prüft ob das Modell geladen ist"""
    return True, "Model is ready"

health.add_check(model_available)
```

## 📈 Skalierung

1. **Horizontale Skalierung**
   - Kubernetes Deployment
   - Load Balancing
   - Auto-Scaling

2. **Vertikale Skalierung**
   - GPU-Unterstützung
   - Memory Optimization
   - Batch Processing

## 🔧 Wartung

1. **Regelmäßige Updates**
   - Dependency Updates
   - Security Patches
   - Model Retraining

2. **Backup-Strategie**
   - Tägliche Backups
   - Disaster Recovery
   - Data Retention 