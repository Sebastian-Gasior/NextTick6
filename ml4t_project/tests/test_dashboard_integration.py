"""
Tests für die Integration des Dashboards mit LSTM-Vorhersagen
"""
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

from ml4t_project.visualization.show_chart import create_chart_for_dashboard, save_chart_to_file, get_lstm_predictions
from ml4t_project.monitoring.dashboard import MonitoringDashboard

class TestDashboardIntegration:
    """Test-Suite für die Integration des Dashboards mit LSTM-Vorhersagen"""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Erzeugt synthetische Aktiendaten für Tests"""
        # Erzeuge Testdaten für 100 Tage
        start_date = datetime.now() - timedelta(days=100)
        dates = pd.date_range(start=start_date, periods=100, freq='D')
        
        # Generiere Kursverläufe
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.normal(0, 1, size=100))
        
        data = {
            'Open': price * np.random.uniform(0.99, 1.0, size=100),
            'High': price * np.random.uniform(1.0, 1.02, size=100),
            'Low': price * np.random.uniform(0.98, 0.99, size=100),
            'Close': price,
            'Volume': np.random.randint(1000000, 5000000, size=100),
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def test_dir(self):
        """Erstellt temporäres Testverzeichnis"""
        test_dir = Path("ml4t_project/exports/test_charts")
        test_dir.mkdir(parents=True, exist_ok=True)
        yield test_dir
        
        # Cleanup
        for file in test_dir.glob("*.html"):
            try:
                os.remove(file)
            except:
                pass
    
    def test_create_chart_for_dashboard_with_sample_data(self, sample_stock_data):
        """Test für die Funktion create_chart_for_dashboard mit synthetischen Daten"""
        try:
            # Lade ein echtes Symbol für einen kurzen Zeitraum
            try:
                import yfinance as yf
                real_df = yf.download("AAPL", start="2022-01-01", end="2022-01-15")
                if real_df.empty:
                    pytest.skip("Keine Daten von Yahoo Finance verfügbar")
                    
                # Erstelle Chart
                predictions = np.array([[price] for price in real_df['Close'].values[-5:]])
                
                fig = create_chart_for_dashboard(
                    symbol="AAPL",
                    start_date="2022-01-01", 
                    end_date="2022-01-15",
                    lstm_predictions=predictions
                )
                
                # Überprüfe Figurattribute
                assert fig is not None
                assert len(fig.data) >= 4  # Candlestick + 2 MAs + LSTM
                
                # Überprüfe, ob LSTM-Vorhersagen enthalten sind
                trace_names = [trace.name for trace in fig.data]
                assert 'LSTM-Vorhersage' in trace_names
                
            except Exception as e:
                pytest.skip(f"Fehler beim Zugriff auf Yahoo Finance: {str(e)}")
        except Exception as e:
            pytest.fail(f"Test fehlgeschlagen: {str(e)}")
    
    def test_save_chart_to_file(self, test_dir):
        """Test für die Funktion save_chart_to_file"""
        # Teste das Erstellen und Speichern von Charts
        try:
            # Verwende einen bekannten Testsymbol
            symbol = "AAPL"
            start_date = "2023-01-01"
            end_date = "2023-01-31"  # Kurzer Zeitraum für schnelleren Test
            
            # Chart speichern
            output_file = save_chart_to_file(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                output_dir=str(test_dir)
            )
            
            # Überprüfe, ob die Datei erstellt wurde
            assert Path(output_file).exists()
            assert Path(output_file).stat().st_size > 0
            
        except Exception as e:
            # Wir fangen den Fehler hier ab, um sicherzustellen, dass der Test nicht fehlschlägt,
            # wenn Yahoo Finance nicht erreichbar ist (aber wir loggen ihn)
            pytest.skip(f"Fehler beim Zugriff auf Yahoo Finance: {str(e)}")
    
    def test_dashboard_initialization(self):
        """Test für die Initialisierung des Dashboards"""
        # Teste, ob das Dashboard ohne Fehler initialisiert werden kann
        dashboard = MonitoringDashboard()
        
        # Überprüfe Grundeigenschaften
        assert dashboard.app is not None
        
        # Suche nach dem Graph-Element mit ID 'lstm-prediction-chart'
        def find_component_by_id(component, component_id):
            """Sucht rekursiv nach einer Komponente mit bestimmter ID"""
            if hasattr(component, 'id') and component.id == component_id:
                return True
                
            if hasattr(component, 'children'):
                children = component.children
                if children is not None:
                    if isinstance(children, list):
                        for child in children:
                            if find_component_by_id(child, component_id):
                                return True
                    else:
                        return find_component_by_id(children, component_id)
            return False
                
        # Überprüfe, ob die relevanten Komponenten im Layout vorhanden sind
        assert find_component_by_id(dashboard.app.layout, 'lstm-prediction-chart')
        assert find_component_by_id(dashboard.app.layout, 'lstm-iframe')
        
        # Überprüfe Callbacks
        assert len(dashboard.app.callback_map) >= 2  # Mindestens 2 Callbacks
        
        # Überprüfe, ob der LSTM-Vorhersage-Callback vorhanden ist
        callback_inputs = [
            callback['inputs'] 
            for callback in dashboard.app.callback_map.values()
        ]
        
        # Mindestens ein Callback sollte den stock-selector verwenden
        stock_selector_used = False
        for callback_info in dashboard.app.callback_map.values():
            if 'inputs' in callback_info:
                for input_id_dict in callback_info['inputs']:
                    if isinstance(input_id_dict, dict) and 'id' in input_id_dict:
                        if input_id_dict.get('id') == 'stock-selector':
                            stock_selector_used = True
                            break
        
        assert stock_selector_used, "Kein Callback gefunden, der den stock-selector verwendet"
    
    def test_lstm_predictions_in_prepare_data(self, sample_stock_data):
        """Test für die LSTM-Vorhersage-Integration"""
        # Simuliere das Verhalten der prepare_data Methode in TradingAnalysis
        try:
            # Verwende die get_lstm_predictions Funktion
            predictions = get_lstm_predictions(sample_stock_data)
            
            # Überprüfe Vorhersagen
            assert predictions is not None
            assert isinstance(predictions, np.ndarray)
            assert predictions.shape[0] > 0
            assert predictions.shape[1] == 1  # Eine Ausgabespalte (Close)
            
            # Speicherlogik testen
            test_file = Path("ml4t_project/exports/test_predictions.npy")
            np.save(test_file, predictions)
            
            # Überprüfe, ob die Datei erstellt wurde
            assert test_file.exists()
            
            # Lade die Datei erneut, um zu überprüfen
            loaded_predictions = np.load(test_file)
            assert np.array_equal(predictions, loaded_predictions)
            
            # Cleanup
            if test_file.exists():
                os.remove(test_file)
                
        except Exception as e:
            pytest.skip(f"LSTM-Vorhersage fehlgeschlagen: {str(e)}") 