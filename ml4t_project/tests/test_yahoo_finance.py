"""
Tests für den Yahoo Finance Datenloader.
"""

import pytest
import torch
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from ml4t_project.data.yahoo_finance import YahooFinanceLoader

@pytest.fixture
def loader():
    """Fixture für YahooFinanceLoader"""
    return YahooFinanceLoader(cache_dir="tests/cache")

@pytest.fixture
def sample_symbol():
    """Fixture für Test-Symbol"""
    return "AAPL"

def test_loader_initialization(loader):
    """Test Loader-Initialisierung"""
    assert isinstance(loader, YahooFinanceLoader)
    assert loader.cache_dir.exists()

def test_data_loading(loader, sample_symbol):
    """Test Datenladen"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = loader.get_data(
        symbol=sample_symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_data_cleaning(loader, sample_symbol):
    """Test Datenbereinigung"""
    data = loader.get_data(sample_symbol)
    
    # Prüfe auf Duplikate
    assert data.index.duplicated().sum() == 0
    
    # Prüfe auf fehlende Werte
    assert data.isnull().sum().sum() == 0
    
    # Prüfe Spaltennamen
    assert all(col.islower() for col in data.columns)

def test_cache_functionality(loader, sample_symbol):
    """Test Cache-Funktionalität"""
    # Erste Ladung
    data1 = loader.get_data(sample_symbol)
    cache_file = loader.cache_dir / f"{sample_symbol}_1d.parquet"
    
    assert cache_file.exists()
    
    # Zweite Ladung (sollte aus Cache kommen)
    data2 = loader.get_data(sample_symbol)
    
    pd.testing.assert_frame_equal(data1, data2)
    
    # Cache löschen
    loader.clear_cache(sample_symbol)
    assert not cache_file.exists()

def test_multiple_symbols(loader):
    """Test Laden mehrerer Symbole"""
    symbols = ['AAPL', 'GOOGL']
    data_dict = loader.get_multiple_symbols(symbols)
    
    assert isinstance(data_dict, dict)
    assert all(symbol in data_dict for symbol in symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data_dict.values())

@pytest.mark.integration
def test_full_pipeline_integration(loader):
    """Integrationstests für den Loader"""
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in symbols:
        # Lade Daten
        data = loader.get_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validiere Daten
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data.index.is_monotonic_increasing
        assert data.index.is_unique
        
        # Prüfe Datenqualität
        assert data.isnull().sum().sum() == 0
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Prüfe Zeitraum
        assert data.index.min() >= pd.to_datetime(start_date)
        assert data.index.max() <= pd.to_datetime(end_date)

@pytest.mark.error
def test_error_handling(loader):
    """Test Fehlerbehandlung"""
    # Test mit ungültigem Symbol
    with pytest.raises(Exception):
        loader.get_data("INVALID_SYMBOL")
    
    # Test mit ungültigem Zeitraum
    future_date = datetime.now() + timedelta(days=30)
    with pytest.raises(Exception):
        loader.get_data("AAPL", start_date=future_date)

def test_cache_validation(loader, sample_symbol):
    """Test Cache-Validierung"""
    # Lade aktuelle Daten
    current_data = loader.get_data(sample_symbol)
    
    # Prüfe Cache-Validität
    assert loader._is_cache_valid(current_data)
    
    # Simuliere alte Daten
    old_data = current_data.copy()
    old_data.index = old_data.index - timedelta(days=2)
    assert not loader._is_cache_valid(old_data)

@pytest.fixture(autouse=True)
def cleanup(loader):
    """Cleanup nach Tests"""
    yield
    loader.clear_cache()  # Lösche alle Cache-Dateien nach den Tests 

class TestYahooFinanceLoader:
    def setup_method(self):
        self.loader = YahooFinanceLoader(
            symbols=['AAPL', 'MSFT'],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            interval='1d'
        )
        
    def test_initialization(self):
        assert self.loader.is_initialized
        assert len(self.loader.symbols) == 2
        assert self.loader.interval == '1d'
        
    def test_data_loading(self):
        data = self.loader.load_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        
    def test_multiple_symbols(self):
        data = self.loader.load_data()
        assert all(symbol in data.index.get_level_values('Symbol') for symbol in self.loader.symbols)
        
    def test_data_validation(self):
        data = self.loader.load_data()
        is_valid = self.loader.validate_data(data)
        assert is_valid
        
    def test_preprocessing(self):
        data = self.loader.load_data()
        processed_data = self.loader.preprocess_data(data)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape[0] > 0
        
    def test_performance_metrics(self):
        metrics = self.loader.get_performance_metrics()
        assert 'loading_speed' in metrics
        assert 'data_points' in metrics
        assert 'memory_usage' in metrics
        
    def test_error_handling(self):
        with pytest.raises(ValueError):
            self.loader.load_data(invalid_param=True)
            
    def test_cleanup(self):
        self.loader.cleanup()
        assert not self.loader.is_initialized 