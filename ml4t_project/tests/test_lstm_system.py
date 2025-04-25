"""
Test des LSTM-Systems für Aktienvorhersagen
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Füge Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ml4t_project.visualization.show_chart import create_chart_for_dashboard, get_lstm_predictions

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lstm_system():
    """Testet das gesamte LSTM-System"""
    try:
        # Test-Parameter
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        logger.info(f"Starte Systemtest für {symbol}")
        
        # Lade Testdaten
        logger.info("Lade Testdaten...")
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} gefunden")
        
        # Setze Symbol-Name für das Training
        df.name = symbol
        
        # Generiere LSTM-Vorhersagen
        logger.info("Generiere LSTM-Vorhersagen...")
        predictions = get_lstm_predictions(df)
        
        # Erstelle normale Ansicht
        logger.info("Erstelle normale Ansicht...")
        fig_normal = create_chart_for_dashboard(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            lstm_predictions=predictions,
            show_signals=True,
            is_detailed=False
        )
        
        # Erstelle detaillierte Ansicht
        logger.info("Erstelle detaillierte Ansicht...")
        fig_detailed = create_chart_for_dashboard(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            lstm_predictions=predictions,
            show_signals=True,
            is_detailed=True
        )
        
        # Speichere Charts
        output_dir = Path("ml4t_project/exports/charts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_normal.write_html(output_dir / f"chart_{symbol}.html")
        fig_detailed.write_html(output_dir / f"chart_{symbol}_detailed.html")
        
        logger.info("Test erfolgreich abgeschlossen!")
        logger.info(f"Charts wurden gespeichert in: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fehler beim Systemtest: {str(e)}")
        return False

if __name__ == "__main__":
    test_lstm_system() 