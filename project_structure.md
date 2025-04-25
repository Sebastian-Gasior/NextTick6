# NextTick6 Projektstruktur v0.8.21

## Hauptverzeichnis

```
NextTick6/
├── .gitignore                    # Git-Ignorier-Konfiguration
├── README.md                     # Hauptprojektdokumentation
├── README_old.md                 # Ältere Version der Dokumentation
├── README_old3.md               # Archivierte Dokumentation
├── requirements.txt              # Python-Abhängigkeiten
├── requirements-dev.txt          # Entwicklungsabhängigkeiten
├── setup.py                      # Python-Paket-Konfiguration
├── pytest.ini                    # PyTest-Konfiguration
├── Dockerfile                    # Docker-Build-Konfiguration
├── docker-compose.yml           # Docker-Compose-Konfiguration
├── trading_config.yaml          # Trading-Konfigurationsdatei
├── integration_plan.md          # Integrationsplanung
├── market_analysis.py           # Hauptmarktanalyse-Modul
├── test_market_analysis.py      # Tests für Marktanalyse
├── master_analysis.py           # Übergeordnete Analysefunktionen
├── show_chart.py               # Diagramm-Visualisierungsskript
├── check_data.py               # Datenvalidierungsskript
└── test_lstm_system.py         # LSTM-System-Tests

## Hauptpaket (ml4t_project/)

```
ml4t_project/
├── __init__.py                 # Paket-Initialisierung
├── config.py                   # Globale Konfiguration
├── main.py                     # Hauptanwendungseinstiegspunkt
├── project-images/            # Projektbilder und Screenshots
├── exports/                   # Exportierte Daten und Modelle
├── models/                    # ML-Modell-Implementierungen
├── analysis/                  # Analysemodule
├── monitoring/                # Dashboard und Überwachung
├── visualization/             # Visualisierungskomponenten
├── processing/               # Datenverarbeitung
├── features/                # Feature Engineering
├── signals/                 # Signalgenerierung
├── backtest/               # Backtesting-System
├── computing/             # Verteiltes Computing
├── optimization/         # Performance-Optimierungen
└── tests/               # Testsuites
```

## Verzeichnisbeschreibungen

### Hauptverzeichnis
- `data/`: Marktdaten und Trainingsdatensätze
- `docs/`: Technische Dokumentation und API-Referenz
- `logs/`: Anwendungs- und Fehlerprotokolle
- `charts/`: Generierte Visualisierungen
- `config/`: Umgebungsspezifische Konfigurationen
- `metrics/`: Performance- und Modellmetriken
- `.venv/`: Python virtuelle Umgebung
- `tests/`: Systemweite Testsuites

### ml4t_project/ (Hauptpaket)
- `models/`: LSTM und andere ML-Modelle
- `analysis/`: Technische und fundamentale Analyse
- `monitoring/`: Echtzeit-Dashboard (Dash)
- `visualization/`: Chart-Generierung und Plotting
- `processing/`: ETL und Datenverarbeitung
- `features/`: Feature Engineering und Indikatoren
- `signals/`: Trading-Signal-Generierung
- `backtest/`: Backtesting-Framework
- `computing/`: Verteiltes Training und GPU-Optimierung
- `optimization/`: Performance-Tuning
- `tests/`: Modulspezifische Tests

## Hauptskripte
- `market_analysis.py`: Implementiert Marktanalyse-Logik
- `master_analysis.py`: Koordiniert verschiedene Analysemodule
- `show_chart.py`: Erzeugt interaktive Visualisierungen
- `check_data.py`: Validiert Datensatzintegrität

## Konfigurationsdateien
- `trading_config.yaml`: Trading-Parameter
- `pytest.ini`: Test-Konfiguration
- `requirements.txt`: Produktionsabhängigkeiten
- `requirements-dev.txt`: Entwicklungsabhängigkeiten

## Docker-Setup
- `Dockerfile`: Container-Build
- `docker-compose.yml`: Multi-Container-Orchestrierung

## Änderungsverlauf v0.8.21

### Neue Features
- Implementierung des LSTM-Modells mit PyTorch
- Integration von verteiltem Training
- Erweiterte Testsuite für ML-Komponenten
- Performance-Optimierungen für Modelltraining
- Docker-Unterstützung für reproduzierbare Entwicklung
- Metriken-Tracking und Performance-Monitoring

### Technologie-Stack
- Python 3.x
- PyTorch für Deep Learning
- Dash für das Dashboard
- Docker für Containerisierung
- PyTest für Tests
- MLflow für Experiment-Tracking