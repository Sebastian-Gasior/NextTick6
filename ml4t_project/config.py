"""
Zentrale Konfigurationsdatei für ML4T
"""
from pathlib import Path
import os
from datetime import datetime, timedelta

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")  # Setze über Umgebungsvariable

# Daten-Parameter
SYMBOL = "AAPL"  # Zu analysierendes Symbol
START_DATE = "2022-01-01"  # Längerer Zeitraum für mehr Daten
END_DATE = "2024-01-01"
TRAIN_TEST_SPLIT = 0.8

# Feature Engineering
WINDOW_SIZE = 20  # Reduziertes Zeitfenster für die verfügbaren Daten
TECHNICAL_INDICATORS = [
    "RSI",
    "MACD",
    "BB_UPPER",
    "BB_LOWER",
    "EMA_20",
]

# Model Parameter
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Signal Parameter
BUY_THRESHOLD = 0.02  # 2% erwartete Rendite für Kauf
SELL_THRESHOLD = -0.02  # -2% erwartete Rendite für Verkauf

# Backtest Parameter
INITIAL_CAPITAL = 10_000.0
POSITION_SIZE = 0.1  # Maximale Positionsgröße (0-1)

# Dateipfade
BASE_PATH = Path(__file__).parent
EXPORTS_PATH = BASE_PATH / "exports"
MODEL_PATH = EXPORTS_PATH / "model.h5"
RESULTS_PATH = EXPORTS_PATH / "results.csv"
PLOT_PATH = EXPORTS_PATH / "performance.html"

# Stelle sicher dass Export-Verzeichnis existiert
EXPORTS_PATH.mkdir(exist_ok=True) 