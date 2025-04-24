# NextTick6 Projektstruktur

```
NextTick6/
├── data/                      # Datenverzeichnis
│   ├── historical/           # Historische Marktdaten
│   │   ├── daily/           # Tägliche Kursdaten
│   │   └── intraday/        # Intraday-Handelsdaten
│   ├── processed/           # Verarbeitete Datensätze
│   │   ├── features/        # Berechnete Features und Indikatoren
│   │   └── indicators/      # Technische Indikatoren
│   ├── raw/                 # Rohdaten von verschiedenen Quellen
│   │   ├── YYYY-MM/        # Monatlich organisierte Rohdaten
│   │   └── latest/         # Aktuelle Marktdaten
│   └── metadata/           # Metadaten und Konfigurationen
├── ml4t_project/           # Hauptprojektverzeichnis
│   ├── analysis/          # Marktanalyse-Module
│   │   ├── market_analyzer.py     # Hauptanalyse-Engine
│   │   └── text_analyzer.py       # Textbasierte Analyse
│   ├── data/             # Datenverarbeitungsmodule
│   │   ├── market_data.py        # Marktdaten-Handler
│   │   ├── data_processor.py     # Datenverarbeitung
│   │   ├── data_validator.py     # Datenvalidierung
│   │   ├── sentiment_data.py     # Sentiment-Daten-Provider
│   │   ├── macro_data.py         # Makroökonomische Daten
│   │   ├── orderflow_data.py     # Order-Flow-Daten
│   │   ├── combined_data_provider.py # Kombinierter Datenprovider
│   │   ├── csv_handler.py        # CSV-Datenhandling
│   │   └── yahoo_finance.py      # Yahoo Finance Integration
│   ├── models/           # ML-Modelle und Training (Konsolidiert)
│   │   ├── lstm_model.py         # LSTM-Implementierung
│   │   ├── model_trainer.py      # Trainings-Engine
│   │   └── model_validator.py    # Modellvalidierung
│   ├── training/         # Training und Optimierung
│   │   ├── distributed_training.py: Framework für verteiltes Training
│   │   ├── checkpoint_manager.py: Checkpoint-Verwaltung
│   │   └── metrics_collector.py: Performance-Metriken
│   ├── monitoring/       # System-Monitoring
│   │   ├── dashboard.py          # Dash-Dashboard
│   │   └── metrics.py            # Performance-Metriken
│   ├── visualization/    # Visualisierungskomponenten
│   │   ├── show_chart.py         # Chart-Generierung mit LSTM-Vorhersagen
│   │   └── __init__.py           # Modul-Initialisierung
│   ├── optimization/     # Leistungsoptimierung
│   │   ├── cache_manager.py      # Cache-Verwaltung
│   │   └── gpu_optimizer.py      # GPU-Optimierung
│   ├── utils/           # Hilfsfunktionen
│   │   ├── config.py            # Konfigurationsmanagement
│   │   └── helpers.py           # Allgemeine Hilfsfunktionen
│   ├── tests/           # Testmodule
│   │   ├── test_market_analyzer.py
│   │   ├── test_data_processor.py
│   │   ├── test_distributed.py         # Tests für verteiltes Training
│   │   ├── test_distributed_training.py # Erweiterte Tests für verteiltes Training
│   │   ├── test_lstm_model.py          # LSTM-Modell Tests
│   │   ├── test_model_trainer.py       # Modelltraining Tests
│   │   ├── test_model_validator.py     # Modellvalidierung Tests
│   │   ├── test_yahoo_integration.py   # Yahoo Finance Integration Tests
│   │   ├── test_monitoring.py          # Monitoring System Tests
│   │   ├── test_visualization.py       # Visualisierungstests
│   │   ├── test_sentiment_data.py      # Sentiment-Daten Tests
│   │   ├── test_macro_data.py          # Makroökonomische Daten Tests
│   │   ├── test_orderflow_data.py      # Order-Flow-Daten Tests
│   │   ├── test_combined_data.py       # Kombinierter Datenprovider Tests
│   │   └── [weitere Testdateien]
│   └── main.py          # Haupteinstiegspunkt der Anwendung
├── master_analysis.py    # Master-Analyse-Skript
├── notebooks/           # Jupyter Notebooks
│   ├── analysis/       # Analyse-Notebooks
│   └── research/       # Forschungs-Notebooks
├── docs/               # Dokumentation
│   ├── api/           # API-Dokumentation
│   ├── user/          # Benutzerhandbuch
│   └── dev/           # Entwicklerdokumentation
├── scripts/           # Hilfsskripte
│   ├── setup.py      # Setup-Skript
│   └── deploy.py     # Deployment-Skript
├── logs/             # Logdateien
├── .git/             # Git-Repository
├── .gitignore        # Git-Ignore-Datei
├── README.md         # Projektbeschreibung
├── project_structure.md # Projektstruktur-Dokumentation
├── trading_config.yaml # Trading-Konfigurationsdatei
├── requirements.txt  # Abhängigkeiten
└── setup.py         # Projekt-Setup
```

## Verzeichniserklärungen

### /data
Enthält alle projektbezogenen Daten in strukturierter Form:
- `historical/`: Historische Marktdaten für Analyse und Training
- `processed/`: Verarbeitete und transformierte Daten für ML-Modelle
- `raw/`: Unverarbeitete Rohdaten von verschiedenen Datenquellen
- `metadata/`: Konfigurationen und Metadaten für Datenverarbeitung

### /ml4t_project
Hauptverzeichnis mit allen Kernmodulen:
- `analysis/`: Module für technische und fundamentale Analyse
  - `market_analyzer.py`: Implementiert die Hauptanalyse-Logik
  - `text_analyzer.py`: Verarbeitet textbasierte Marktdaten
- `data/`: Datenverarbeitung und -management
  - `market_data.py`: Lädt und verwaltet Marktdaten
  - `data_processor.py`: Verarbeitet und transformiert Daten
  - `data_validator.py`: Validiert Datenqualität
  - `sentiment_data.py`: Provider für Sentiment-Daten (News, Social Media)
  - `macro_data.py`: Provider für makroökonomische Daten
  - `orderflow_data.py`: Provider für Order-Flow-Daten
  - `combined_data_provider.py`: Kombiniert verschiedene Datenquellen
  - `csv_handler.py`: Utilities für CSV-Dateihandling
  - `yahoo_finance.py`: Integration mit Yahoo Finance API
- `models/`: ML-Modelle und Trainingslogik (Konsolidiert)
  - `lstm_model.py`: LSTM-Modell für Zeitreihenvorhersage
  - `model_trainer.py`: Training und Optimierung
  - `model_validator.py`: Modellvalidierung und -tests
- `training/`: Training und Optimierung
  - `distributed_training.py`: Framework für verteiltes Training
  - `checkpoint_manager.py`: Checkpoint-Verwaltung
  - `metrics_collector.py`: Performance-Metriken
- `visualization/`: Visualisierungskomponenten
  - `show_chart.py`: Generiert interaktive Charts mit LSTM-Vorhersagen
  - `__init__.py`: Modul-Initialisierung
- `monitoring/`: System- und Performance-Monitoring
  - `dashboard.py`: Interaktives Dash-Dashboard
  - `metrics.py`: Performance- und Systemmetriken
- `optimization/`: Performance-Optimierungen
  - `cache_manager.py`: Intelligentes Caching-System
  - `gpu_optimizer.py`: GPU-basierte Beschleunigung
- `utils/`: Hilfsfunktionen und Tools
  - `config.py`: Konfigurationsmanagement
  - `helpers.py`: Allgemeine Hilfsfunktionen
- `tests/`: Automatisierte Tests für alle Module
- `main.py`: Haupteinstiegspunkt der Anwendung

### /master_analysis.py
Master-Skript für die Trading-Analyse:
- Initialisiert die Komponenten
- Lädt Daten für alle konfigurierten Symbole
- Generiert Analysen und Metriken
- Startet das Monitoring-Dashboard

### /notebooks
Jupyter Notebooks für Analyse und Forschung:
- `analysis/`: Detaillierte Marktanalysen und Visualisierungen
- `research/`: Forschung und Experimente zu neuen Features

### /docs
Umfassende Projektdokumentation:
- `api/`: API-Referenz und Schnittstellendokumentation
- `user/`: Benutzerhandbuch und Tutorials
- `dev/`: Entwicklerdokumentation und Beitragsrichtlinien

### /scripts
Hilfsskripte für verschiedene Aufgaben:
- `setup.py`: Projekt-Setup und Initialisierung
- `deploy.py`: Deployment-Automatisierung

### /logs
Logdateien für Debugging und Monitoring
- Enthält strukturierte Logs für System und Performance
- Automatische Log-Rotation und Archivierung

## Änderungshistorie

### 2024-06-01: Implementierung von Sentiment- und Makrodaten
- Hinzufügung des Sentiment-Daten-Providers
- Hinzufügung des Makroökonomischen Daten-Providers
- Hinzufügung des Order-Flow-Daten-Providers
- Implementierung des kombinierten Datenproviders
- Hinzufügung der entsprechenden Tests

### 2024-05-15: Hinzufügung der Visualisierungskomponente
- Hinzufügung der `visualization/`-Komponente
- Implementation von `show_chart.py` für LSTM-Vorhersagevisualisierung
- Tests für die Visualisierungskomponente
- Integration in das Hauptsystem

### 2024-04-24: Konsolidierung der Modellimplementierungen
- 24.04.2024: Konsolidierung der Modellimplementierungen
- 24.04.2024: Implementierung der Modellvalidierung mit synthetischen Daten
- 24.04.2024: Korrektur der LSTM-Modellausgabe
- 24.04.2024: Anpassung des ModelValidator
- 24.04.2024: Integration von Yahoo Finance für Echtzeit-Daten
- 24.04.2024: Implementierung der Datenpipeline und Vorverarbeitung
- 24.04.2024: Korrektur der Datetime-Verarbeitung (UTC)
- 24.04.2024: Korrektur der MA-Analyse
- 24.04.2024: Aktualisierung der Tests und Dokumentation

### 2024-04-24: Implementierung des Distributed Training Frameworks
- Implementierung des verteilten Trainings
- Worker-Node-Management
- Task-Verteilung und Synchronisation
- Checkpoint-System
- Performance-Metriken
- Integration der LSTM-Modelle für Aktienvorhersage
- Implementierung umfangreicher Testsuite für verteiltes Training
- Optimierung der Modellvalidierung