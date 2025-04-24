"""
Tests für den kombinierten Daten-Provider
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml4t_project.data.combined_data_provider import CombinedDataProvider

class TestCombinedDataProvider:
    """Test-Suite für den kombinierten Daten-Provider"""
    
    @pytest.fixture
    def data_provider(self):
        """Fixture für den kombinierten Daten-Provider"""
        return CombinedDataProvider()
    
    @pytest.fixture
    def test_dates(self):
        """Fixture für Testdaten"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 Monate
        return start_date, end_date
    
    def test_get_complete_data(self, data_provider, test_dates):
        """Test für get_complete_data"""
        start_date, end_date = test_dates
        
        # Lade vollständige Daten für ein bekanntes Symbol
        symbol = 'AAPL'
        data = data_provider.get_complete_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            include_sentiment=True,
            include_macro=True,
            include_orderflow=True,
            include_technical_indicators=True
        )
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        
        # Prüfe verschiedene Datenquellen
        # 1. Marktdaten
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        
        # 2. Technische Indikatoren
        assert 'MA_20' in data.columns
        assert 'RSI' in data.columns
        assert 'MACD' in data.columns
        
        # 3. Order Flow
        flow_columns = [col for col in data.columns if col.startswith('flow_')]
        assert len(flow_columns) > 0
        
        # 4. Sentiment (bei synthetischen Daten)
        sentiment_columns = [col for col in data.columns if col.startswith('sentiment_')]
        assert len(sentiment_columns) > 0
        
        # 5. Makro
        macro_columns = [col for col in data.columns if col.startswith('macro_')]
        assert len(macro_columns) > 0
        
        # Prüfe auf NaN-Werte (sollten keine sein)
        assert not data.isna().any().any()
        
    def test_separate_data_sources(self, data_provider, test_dates):
        """Test für das Laden separater Datenquellen"""
        start_date, end_date = test_dates
        symbol = 'MSFT'
        
        # 1. Nur Marktdaten
        market_only = data_provider.get_complete_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            include_sentiment=False,
            include_macro=False,
            include_orderflow=False,
            include_technical_indicators=False
        )
        
        assert market_only is not None
        assert isinstance(market_only, pd.DataFrame)
        assert not market_only.empty
        
        # Prüfe, dass nur Marktdaten vorhanden sind
        assert 'Open' in market_only.columns
        assert 'Close' in market_only.columns
        
        sentiment_columns = [col for col in market_only.columns if col.startswith('sentiment_')]
        assert len(sentiment_columns) == 0
        
        macro_columns = [col for col in market_only.columns if col.startswith('macro_')]
        assert len(macro_columns) == 0
        
        flow_columns = [col for col in market_only.columns if col.startswith('flow_')]
        assert len(flow_columns) == 0
        
        # 2. Marktdaten + Technische Indikatoren
        tech_data = data_provider.get_complete_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            include_sentiment=False,
            include_macro=False,
            include_orderflow=False,
            include_technical_indicators=True
        )
        
        assert tech_data is not None
        assert isinstance(tech_data, pd.DataFrame)
        assert not tech_data.empty
        
        # Prüfe technische Indikatoren
        assert 'MA_20' in tech_data.columns
        assert 'RSI' in tech_data.columns
        
    def test_prepare_lstm_data(self, data_provider, test_dates):
        """Test für prepare_lstm_data"""
        start_date, end_date = test_dates
        symbol = 'GOOGL'
        
        # Lade Daten
        data = data_provider.get_complete_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Bereite LSTM-Daten vor
        lstm_data = data_provider.prepare_lstm_data(
            df=data,
            target_column='Close',
            sequence_length=10,
            prediction_horizon=1,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            scaling=True
        )
        
        # Prüfe, ob ein Dictionary zurückgegeben wurde
        assert lstm_data is not None
        assert isinstance(lstm_data, dict)
        
        # Prüfe Schlüssel
        assert 'X_train' in lstm_data
        assert 'y_train' in lstm_data
        assert 'X_val' in lstm_data
        assert 'y_val' in lstm_data
        assert 'X_test' in lstm_data
        assert 'y_test' in lstm_data
        assert 'feature_scaler' in lstm_data
        assert 'target_scaler' in lstm_data
        assert 'features' in lstm_data
        assert 'target_column' in lstm_data
        
        # Prüfe Formen der Arrays
        X_train = lstm_data['X_train']
        y_train = lstm_data['y_train']
        
        # Shape sollte sein: (Samples, Sequence Length, Features)
        assert len(X_train.shape) == 3
        assert X_train.shape[1] == 10  # Sequence Length
        
        # Shape von y sollte sein: (Samples, Target Features)
        assert len(y_train.shape) == 2
        assert y_train.shape[1] == 1  # Ein Target
        
        # Prüfe, dass X_train und y_train gleich viele Samples haben
        assert X_train.shape[0] == y_train.shape[0]
        
    def test_train_val_test_split(self, data_provider):
        """Test für get_train_val_test_split"""
        symbol = 'AMZN'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 Monate
        
        # Lade und teile Daten
        split_data = data_provider.get_train_val_test_split(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            separate_validation_period=True
        )
        
        # Prüfe, ob ein Dictionary zurückgegeben wurde
        assert split_data is not None
        assert isinstance(split_data, dict)
        
        # Prüfe Schlüssel
        assert 'train' in split_data
        assert 'validation' in split_data
        assert 'test' in split_data
        
        # Prüfe, dass jeder Datensatz ein DataFrame ist
        assert isinstance(split_data['train'], pd.DataFrame)
        assert isinstance(split_data['validation'], pd.DataFrame)
        assert isinstance(split_data['test'], pd.DataFrame)
        
        # Prüfe, dass die Datensätze nicht leer sind
        assert not split_data['train'].empty
        assert not split_data['validation'].empty
        assert not split_data['test'].empty
        
        # Prüfe, dass die Datensätze zeitlich getrennt sind
        assert split_data['train'].index.max() < split_data['validation'].index.min()
        assert split_data['validation'].index.max() < split_data['test'].index.min()
        
    def test_error_handling(self, data_provider):
        """Test für Fehlerbehandlung"""
        # Test mit ungültigem Symbol
        data = data_provider.get_complete_data(
            symbol='INVALID_SYMBOL',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Sollte None zurückgeben
        assert data is None
        
        # Test mit leerem DataFrame für prepare_lstm_data
        empty_df = pd.DataFrame()
        lstm_data = data_provider.prepare_lstm_data(empty_df)
        
        # Sollte leeres Dictionary zurückgeben
        assert isinstance(lstm_data, dict)
        assert len(lstm_data) == 0 