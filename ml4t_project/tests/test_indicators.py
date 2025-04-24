"""
Test-Modul für technische Indikatoren
"""
import pytest
import pandas as pd
import numpy as np
from ml4t_project.features.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_ema

@pytest.fixture
def sample_price_data():
    """Beispiel-Preisdaten für Tests"""
    # Generiere 30 Tage Testdaten
    np.random.seed(42)  # Für reproduzierbare Tests
    n_days = 30
    
    # Generiere realistische Preisbewegungen
    close = 100.0
    closes = [close]
    for _ in range(n_days - 1):
        change = np.random.normal(0, 1)  # Zufällige Preisänderung
        close = close * (1 + change/100)  # Prozentuale Änderung
        closes.append(close)
    
    closes = np.array(closes)
    
    return pd.DataFrame({
        'Close': closes,
        'High': closes * 1.01,  # 1% über Close
        'Low': closes * 0.99,   # 1% unter Close
    })

def test_rsi_calculation(sample_price_data):
    """Test der RSI-Berechnung"""
    # Setup
    period = 14
    
    # Ausführung
    rsi = calculate_rsi(sample_price_data['Close'], period)
    
    # Assertions
    assert isinstance(rsi, pd.Series), "RSI sollte als Series zurückgegeben werden"
    assert len(rsi) == len(sample_price_data), "RSI-Länge sollte Datenlänge entsprechen"
    assert all(0 <= value <= 100 for value in rsi.dropna()), "RSI-Werte sollten zwischen 0 und 100 liegen"
    assert rsi.iloc[:period-1].isna().all(), f"Erste {period-1} Werte sollten NaN sein"

def test_macd_calculation(sample_price_data):
    """Test der MACD-Berechnung"""
    # Setup
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # Ausführung
    macd_line, signal_line, macd_hist = calculate_macd(
        sample_price_data['Close'],
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period
    )
    
    # Assertions
    assert isinstance(macd_line, pd.Series), "MACD Line sollte Series sein"
    assert isinstance(signal_line, pd.Series), "Signal Line sollte Series sein"
    assert isinstance(macd_hist, pd.Series), "MACD Histogram sollte Series sein"
    assert len(macd_line) == len(sample_price_data), "MACD-Länge sollte Datenlänge entsprechen"
    assert macd_line.iloc[:slow_period-1].isna().all(), f"Erste {slow_period-1} MACD-Werte sollten NaN sein"

def test_bollinger_bands(sample_price_data):
    """Test der Bollinger Bands Berechnung"""
    # Setup
    period = 20
    std_dev = 2
    
    # Ausführung
    upper_band, middle_band, lower_band = calculate_bollinger_bands(
        sample_price_data['Close'],
        period=period,
        std_dev=std_dev
    )
    
    # Assertions
    assert isinstance(upper_band, pd.Series), "Upper Band sollte Series sein"
    assert isinstance(middle_band, pd.Series), "Middle Band sollte Series sein"
    assert isinstance(lower_band, pd.Series), "Lower Band sollte Series sein"
    assert all(upper_band >= middle_band), "Upper Band sollte über Middle Band liegen"
    assert all(lower_band <= middle_band), "Lower Band sollte unter Middle Band liegen"

def test_ema_calculation(sample_price_data):
    """Test der EMA-Berechnung"""
    # Setup
    period = 20
    
    # Ausführung
    ema = calculate_ema(sample_price_data['Close'], period)
    
    # Assertions
    assert isinstance(ema, pd.Series), "EMA sollte Series sein"
    assert len(ema) == len(sample_price_data), "EMA-Länge sollte Datenlänge entsprechen"
    assert ema.iloc[:period-1].isna().all(), f"Erste {period-1} EMA-Werte sollten NaN sein"

def test_indicators_with_insufficient_data():
    """Test der Indikatoren mit zu wenig Daten"""
    # Setup
    short_data = pd.Series([100.0, 101.0, 102.0])
    
    # Assertions
    with pytest.raises(ValueError):
        calculate_rsi(short_data, period=14)
    
    with pytest.raises(ValueError):
        calculate_macd(short_data)
        
    with pytest.raises(ValueError):
        calculate_bollinger_bands(short_data)
        
    with pytest.raises(ValueError):
        calculate_ema(short_data, period=20)

def test_indicators_with_invalid_data():
    """Test der Indikatoren mit ungültigen Daten"""
    # Setup
    invalid_data = pd.Series([100.0, np.nan, 102.0])
    
    # Assertions
    with pytest.raises(ValueError):
        calculate_rsi(invalid_data, period=2)
    
    with pytest.raises(ValueError):
        calculate_macd(invalid_data)
        
    with pytest.raises(ValueError):
        calculate_bollinger_bands(invalid_data)
        
    with pytest.raises(ValueError):
        calculate_ema(invalid_data, period=2)

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 