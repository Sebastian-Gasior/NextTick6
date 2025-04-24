"""
Master-Script für die ML4T Trading-Analyse
"""
import sys
import os
from pathlib import Path
import threading
import webbrowser
from time import sleep
import logging
import yaml
import numpy as np

# Interne Imports
from ml4t_project.monitoring.dashboard import MonitoringDashboard
from ml4t_project.data.market_data import MarketData
from ml4t_project.analysis.market_analyzer import MarketAnalyzer
from ml4t_project.visualization.show_chart import get_lstm_predictions

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingAnalysis:
    """Hauptklasse für die Trading-Analyse"""
    
    def __init__(self):
        self.config_file = "trading_config.yaml"
        self.symbols = []
        self.start_date = ""
        self.end_date = ""
        self.load_config()
        
        # Erstelle Verzeichnisse
        self.setup_directories()
        
        # Initialisiere Komponenten
        self.market_data = MarketData()
        self.market_analyzer = MarketAnalyzer()
        self.monitoring_dashboard = None
        
        # Server-Port
        self.dash_port = 8050
        
    def setup_directories(self):
        """Erstellt benötigte Verzeichnisse"""
        dirs = [
            "ml4t_project/exports",
            "ml4t_project/exports/analysis",
            "ml4t_project/exports/charts",
            "logs"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
    def load_config(self):
        """Lade Konfiguration aus YAML-Datei"""
        if not os.path.exists(self.config_file):
            # Erstelle Standard-Konfiguration
            default_config = {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
                'date_range': {
                    'start': '2022-01-01',
                    'end': '2024-01-01'
                }
            }
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f)
            
        # Lade Konfiguration
        with open(self.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        self.symbols = config_data['symbols']
        self.start_date = config_data['date_range']['start']
        self.end_date = config_data['date_range']['end']
        
    def prepare_data(self):
        """Bereitet die Daten für alle Symbole vor"""
        logger.info("Lade Marktdaten...")
        
        for symbol in self.symbols:
            try:
                # Lade und verarbeite Daten
                df = self.market_data.get_stock_data(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if df is not None:
                    # Generiere Analyse direkt
                    analysis = self.market_analyzer.analyze_stock(df, symbol)
                    logger.info(f"Analyse für {symbol} generiert")
                    
                    # Berechne Metriken
                    metrics, signals = self.market_data.calculate_metrics(df)
                    
                    # Speichere Metriken
                    metrics_file = Path(f"ml4t_project/exports/metrics_{symbol}.json")
                    signals_file = Path(f"ml4t_project/exports/signals_{symbol}.json")
                    
                    import json
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f)
                    with open(signals_file, 'w') as f:
                        json.dump(signals, f)
                        
                    # Generiere LSTM-Vorhersagen
                    try:
                        logger.info(f"Generiere LSTM-Vorhersagen für {symbol}...")
                        predictions = get_lstm_predictions(df)
                        
                        # Speichere Vorhersagen
                        predictions_file = Path(f"ml4t_project/exports/lstm_predictions_{symbol}.npy")
                        np.save(predictions_file, predictions)
                        logger.info(f"LSTM-Vorhersagen für {symbol} generiert und gespeichert")
                    except Exception as e:
                        logger.error(f"Fehler bei LSTM-Vorhersagen für {symbol}: {str(e)}")
                    
                    logger.info(f"Daten für {symbol} erfolgreich verarbeitet")
                    
            except Exception as e:
                logger.error(f"Fehler bei der Datenverarbeitung für {symbol}: {str(e)}")
                continue
    
    def start_dashboard(self):
        """Startet das Trading-Dashboard"""
        try:
            logger.info(f"Starte Dashboard auf Port {self.dash_port}...")
            
            # Bereite Daten vor
            self.prepare_data()
            
            # Stelle sicher, dass das Charts-Verzeichnis existiert
            charts_dir = Path("ml4t_project/exports/charts")
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Starte Dashboard
            self.monitoring_dashboard = MonitoringDashboard()
            
            # Öffne Browser
            webbrowser.open(f'http://localhost:{self.dash_port}')
            
            # Starte Dashboard
            self.monitoring_dashboard.run(debug=False, port=self.dash_port)
            
        except Exception as e:
            logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")
            raise

def main():
    """Hauptfunktion"""
    try:
        # Starte Analyse
        analysis = TradingAnalysis()
        analysis.start_dashboard()
        
    except KeyboardInterrupt:
        logger.info("\nAnalyse beendet.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 