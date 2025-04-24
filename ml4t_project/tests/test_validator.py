import pytest
import pandas as pd
import numpy as np
from ml4t_project.data.validator import DataValidator

@pytest.fixture
def valid_ohlcv_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

@pytest.fixture
def invalid_ohlcv_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'extra_column': [1, 2, 3, 4, 5]  # Zusätzliche Spalte
    })

def test_validate_ohlcv_data(valid_ohlcv_data, invalid_ohlcv_data):
    validator = DataValidator()
    
    # Teste gültige Daten
    assert validator.validate_ohlcv_data(valid_ohlcv_data)
    
    # Teste ungültige Daten
    assert not validator.validate_ohlcv_data(invalid_ohlcv_data)

def test_validate_ohlcv_data_missing_columns():
    validator = DataValidator()
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0]
        # volume fehlt
    })
    assert not validator.validate_ohlcv_data(data)

def test_validate_ohlcv_data_invalid_types():
    validator = DataValidator()
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'open': ['100.0', '101.0', '102.0', '103.0', '104.0'],  # String statt float
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    assert not validator.validate_ohlcv_data(data)

def test_validate_ohlcv_data_invalid_values():
    validator = DataValidator()
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, -1200, 1300, 1400]  # Negatives Volumen
    })
    assert not validator.validate_ohlcv_data(data)

def test_valid_timestamps(validator, valid_ohlcv_data):
    assert validator.validate_timestamps(valid_ohlcv_data)

def test_duplicate_timestamps(validator, valid_ohlcv_data):
    # Füge doppelten Zeitstempel hinzu
    invalid_data = valid_ohlcv_data.copy()
    invalid_data.loc[5] = invalid_data.loc[0]
    assert not validator.validate_timestamps(invalid_data)

def test_non_chronological_timestamps(validator, valid_ohlcv_data):
    # Mache Zeitstempel nicht chronologisch
    invalid_data = valid_ohlcv_data.copy()
    invalid_data.loc[0, 'timestamp'] = pd.Timestamp('2024-01-10')
    assert not validator.validate_timestamps(invalid_data)

def test_missing_timestamps(validator, valid_ohlcv_data):
    # Füge fehlenden Zeitstempel hinzu
    invalid_data = valid_ohlcv_data.copy()
    invalid_data.loc[0, 'timestamp'] = pd.NaT
    assert not validator.validate_timestamps(invalid_data) 