"""
Erweiterte Tests für den Yahoo Finance Datenloader.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from ..data.data_loader import YahooFinanceLoader
import torch
from ml4t_project.data.data_loader import DataLoader

@pytest.fixture
def loader():
    """Fixture für YahooFinanceLoader"""
    return YahooFinanceLoader(cache_dir="tests/cache")

@pytest.fixture
def sample_symbols():
    """Fixture für Test-Symbole"""
    return ['AAPL', 'GOOGL', 'MSFT']

def test_basic_data_loading(loader, sample_symbols):
    """Test grundlegendes Datenladen"""
    data = loader.get_data(sample_symbols[0])
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.parametrize("interval", ['1d', '1h', '1wk', '1mo'])
def test_different_intervals(loader, sample_symbols, interval):
    """Test verschiedene Zeitintervalle"""
    data = loader.get_data(sample_symbols[0], interval=interval)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert data.index.freq is not None or interval == '1d'  # 1d hat keine explizite Frequenz

def test_date_range_validation(loader, sample_symbols):
    """Test Datumsbereich-Validierung"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = loader.get_data(
        sample_symbols[0],
        start_date=start_date,
        end_date=end_date
    )
    
    assert not data.empty
    assert data.index.min().date() >= start_date.date()
    assert data.index.max().date() <= end_date.date()

def test_cache_mechanism(loader, sample_symbols):
    """Test Cache-Mechanismus"""
    symbol = sample_symbols[0]
    
    # Erste Anfrage
    data1 = loader.get_data(symbol)
    cache_file = loader.cache_dir / f"{symbol}_1d.parquet"
    
    assert cache_file.exists()
    
    # Zweite Anfrage (sollte Cache nutzen)
    data2 = loader.get_data(symbol)
    pd.testing.assert_frame_equal(data1, data2)

def test_multiple_symbols(loader, sample_symbols):
    """Test Laden mehrerer Symbole"""
    data_dict = loader.get_multiple_symbols(sample_symbols)
    
    assert isinstance(data_dict, dict)
    assert all(symbol in data_dict for symbol in sample_symbols)
    assert all(isinstance(df, pd.DataFrame) for df in data_dict.values())

@pytest.mark.error
def test_error_handling(loader):
    """Test Fehlerbehandlung"""
    # Test mit ungültigem Symbol
    with pytest.raises(Exception):
        loader.get_data("INVALID_SYMBOL_123")
    
    # Test mit ungültigem Zeitraum
    with pytest.raises(Exception):
        future_date = datetime.now() + timedelta(days=30)
        loader.get_data("AAPL", start_date=future_date)
    
    # Test mit ungültigem Intervall
    with pytest.raises(ValueError):
        loader.get_data("AAPL", interval="invalid")

def test_data_validation(loader, sample_symbols):
    """Test Datenvalidierung"""
    data = loader.get_data(sample_symbols[0])
    
    # Prüfe Datentypen
    assert data.index.dtype == 'datetime64[ns]'
    assert all(data[col].dtype in ['float64', 'int64'] for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Prüfe Wertebereich
    assert all(data['low'] <= data['high'])
    assert all(data['volume'] >= 0)

@pytest.mark.parametrize("column", ['open', 'high', 'low', 'close', 'volume'])
def test_missing_data_handling(loader, sample_symbols, column):
    """Test Behandlung fehlender Daten"""
    data = loader.get_data(sample_symbols[0])
    
    # Simuliere fehlende Werte
    data.loc[data.index[0], column] = np.nan
    
    # Prüfe Behandlung
    cleaned_data = loader._clean_data(data)
    assert not cleaned_data[column].isna().any()

@pytest.mark.integration
def test_full_pipeline(loader, sample_symbols):
    """Integrationstests für die gesamte Pipeline"""
    for symbol in sample_symbols:
        # Teste verschiedene Zeiträume
        for days in [7, 30, 90]:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = loader.get_data(
                symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            assert not data.empty
            assert data.index.is_monotonic_increasing
            assert not data.index.has_duplicates
            assert not data.isna().any().any()

@pytest.mark.performance
def test_cache_performance(loader, sample_symbols):
    """Test Cache-Performance"""
    symbol = sample_symbols[0]
    
    # Erste Anfrage (ohne Cache)
    start_time = datetime.now()
    _ = loader.get_data(symbol)
    first_request_time = (datetime.now() - start_time).total_seconds()
    
    # Zweite Anfrage (mit Cache)
    start_time = datetime.now()
    _ = loader.get_data(symbol)
    cached_request_time = (datetime.now() - start_time).total_seconds()
    
    # Cache sollte deutlich schneller sein
    assert cached_request_time < first_request_time

@pytest.fixture(autouse=True)
def cleanup(loader):
    """Cleanup nach Tests"""
    yield
    import shutil
    try:
        shutil.rmtree(loader.cache_dir)
    except:
        pass

class TestDataLoader:
    def setup_method(self):
        self.loader = DataLoader(
            batch_size=1000,
            num_workers=4,
            prefetch_factor=2
        )
        
    def test_initialization(self):
        assert self.loader.is_initialized
        assert self.loader.batch_size == 1000
        assert self.loader.num_workers == 4
        
    def test_data_loading(self):
        data = self.loader.load_data("test_data.csv")
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'timestamp' in data.columns
        
    def test_batch_processing(self):
        data = torch.randn(1000, 10)
        batches = self.loader.process_batches(data)
        assert len(batches) == 1
        assert batches[0].shape == (1000, 10)
        
    def test_preprocessing(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100),
            'value': range(100)
        })
        
        processed_data = self.loader.preprocess_data(data)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape[0] == 100
        
    def test_validation(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100),
            'value': range(100)
        })
        
        is_valid = self.loader.validate_data(data)
        assert is_valid
        
    def test_error_handling(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_data("non_existent_file.csv")
            
    def test_performance_metrics(self):
        metrics = self.loader.get_performance_metrics()
        assert 'loading_speed' in metrics
        assert 'processing_speed' in metrics
        assert 'memory_usage' in metrics
        
    def test_cleanup(self):
        self.loader.cleanup()
        assert not self.loader.is_initialized 