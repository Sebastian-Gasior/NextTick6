# ML4T Deployment Guide

## Voraussetzungen

- Docker & Docker Compose
- Git
- Python 3.9+
- uv (Python Paketmanager)

## 1. Projekt-Setup

1. Repository klonen:
```bash
git clone https://github.com/Sebastian-Gasior/ml4t_project.git
cd ml4t_project
```

2. Python-Umgebung erstellen:
```bash
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# oder
.venv\Scripts\activate  # Windows
```

3. Abhängigkeiten installieren:
```bash
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## 2. Konfiguration

1. Umgebungsvariablen setzen:
```bash
cp .env.example .env
# Bearbeiten Sie .env mit Ihren Einstellungen
```

2. Monitoring-Konfiguration anpassen:
```bash
# Bearbeiten Sie config/monitoring.yaml
# Bearbeiten Sie config/prometheus.yml
# Bearbeiten Sie config/alert.rules
```

## 3. Docker-Deployment

1. Images bauen:
```bash
docker-compose build
```

2. Services starten:
```bash
docker-compose up -d
```

3. Status überprüfen:
```bash
docker-compose ps
docker-compose logs -f
```

## 4. Monitoring-Setup

1. Grafana öffnen:
- URL: http://localhost:3000
- Standard-Login:
  - Benutzer: admin
  - Passwort: admin

2. Prometheus-Datenquelle hinzufügen:
- URL: http://prometheus:9090
- Access: Server (default)

3. Dashboard importieren:
- Dashboard ID: ML4T-Dashboard
- Datenquelle: Prometheus

## 5. Skalierung

1. Horizontal skalieren:
```bash
docker-compose up -d --scale ml4t=3
```

2. Ressourcen anpassen:
- Bearbeiten Sie die `deploy` Sektion in `docker-compose.yml`

## 6. Wartung

1. Logs überprüfen:
```bash
docker-compose logs -f [service]
```

2. Services aktualisieren:
```bash
docker-compose pull
docker-compose up -d
```

3. Backup erstellen:
```bash
docker-compose exec prometheus sh -c "tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /prometheus"
docker-compose exec grafana sh -c "tar czf /backup/grafana-$(date +%Y%m%d).tar.gz /var/lib/grafana"
```

## 7. Monitoring

### Metriken
- CPU-Auslastung
- Speicherverbrauch
- GPU-Nutzung
- Festplattennutzung
- Latenzzeiten
- Fehlerraten

### Alerts
- CPU > 80%
- Speicher > 80%
- GPU > 80%
- Festplatte > 80%
- Fehlerrate > 5%

## 8. Troubleshooting

1. Container-Logs prüfen:
```bash
docker-compose logs -f [service]
```

2. Service neustarten:
```bash
docker-compose restart [service]
```

3. Alle Services neustarten:
```bash
docker-compose down
docker-compose up -d
```

4. Volumes prüfen:
```bash
docker volume ls
docker volume inspect [volume]
```

5. Netzwerk prüfen:
```bash
docker network ls
docker network inspect ml4t_default
```

## 9. Backup & Recovery

1. Volumes sichern:
```bash
docker run --rm -v ml4t_prometheus_data:/data -v $(pwd)/backup:/backup \
    ubuntu tar czf /backup/prometheus-data.tar.gz /data

docker run --rm -v ml4t_grafana_data:/data -v $(pwd)/backup:/backup \
    ubuntu tar czf /backup/grafana-data.tar.gz /data
```

2. Volumes wiederherstellen:
```bash
docker run --rm -v ml4t_prometheus_data:/data -v $(pwd)/backup:/backup \
    ubuntu bash -c "cd /data && tar xzf /backup/prometheus-data.tar.gz --strip 1"

docker run --rm -v ml4t_grafana_data:/data -v $(pwd)/backup:/backup \
    ubuntu bash -c "cd /data && tar xzf /backup/grafana-data.tar.gz --strip 1"
```

## 10. Security

1. Netzwerk-Sicherheit:
- Verwenden Sie ein privates Docker-Netzwerk
- Beschränken Sie die exponierten Ports
- Aktivieren Sie TLS für externe Verbindungen

2. Container-Sicherheit:
- Regelmäßige Updates
- Keine root-Benutzer
- Read-only Filesystems wo möglich

3. Monitoring-Sicherheit:
- Ändern Sie Standard-Passwörter
- Aktivieren Sie Authentifizierung
- Beschränken Sie den Zugriff auf Metriken

## 11. Performance-Optimierung

1. Container-Optimierung:
- Ressourcen-Limits setzen
- Multi-stage Builds verwenden
- Layer-Caching optimieren

2. Monitoring-Optimierung:
- Scrape-Intervall anpassen
- Retention-Policies setzen
- Dashboard-Caching aktivieren

## 12. Produktions-Checkliste

- [ ] Alle Passwörter geändert
- [ ] Backups eingerichtet
- [ ] Monitoring aktiv
- [ ] Alerts konfiguriert
- [ ] SSL/TLS aktiviert
- [ ] Logging eingerichtet
- [ ] Skalierung getestet
- [ ] Recovery-Prozeduren dokumentiert
- [ ] Security-Scans durchgeführt
- [ ] Performance-Tests abgeschlossen 