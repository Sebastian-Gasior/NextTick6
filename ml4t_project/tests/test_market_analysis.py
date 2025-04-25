"""
import pytest
import pandas as pd
import numpy as np
from market_analysis import MarktAnalyse
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ticker():
    with patch('yfinance.Ticker') as mock:
        # Mock für Fundamentaldaten
        mock.return_value.info = {
            'longName': 'Test Company',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 1000000000,
            'trailingPE': 20.5,
            'dividendYield': 0.02
        }
        
        # Mock für historische Daten
        hist_data = pd.DataFrame({
            'Close': np.linspace(100, 200, 100),
            'Open': np.linspace(98, 198, 100),
            'High': np.linspace(102, 202, 100),
            'Low': np.linspace(97, 197, 100),
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=pd.date_range(start='2023-01-01', periods=100))
        mock.return_value.history.return_value = hist_data
        
        yield mock

def test_initialisierung():
    analyse = MarktAnalyse("TEST")
    assert analyse.symbol == "TEST"

def test_hole_fundamentaldaten(mock_ticker):
    analyse = MarktAnalyse("TEST")
    daten = analyse.hole_fundamentaldaten()
    
    assert isinstance(daten, dict)
    assert daten['Name'] == 'Test Company'
    assert daten['Sektor'] == 'Technology'
    assert daten['Marktkapitalisierung'] == 1000000000

def test_technische_analyse(mock_ticker):
    analyse = MarktAnalyse("TEST")
    df = analyse.technische_analyse('1y')
    
    assert isinstance(df, pd.DataFrame)
    assert 'SMA_20' in df.columns
    assert 'SMA_50' in df.columns
    assert 'RSI' in df.columns
    assert 'MACD' in df.columns
    
    # Überprüfe, ob die technischen Indikatoren korrekt berechnet wurden
    assert not df['SMA_20'].isna().all()
    assert not df['SMA_50'].isna().all()
    assert not df['RSI'].isna().all()
    assert not df['MACD'].isna().all()

def test_berechne_rsi():
    analyse = MarktAnalyse("TEST")
    preise = pd.Series(np.linspace(100, 200, 100))
    rsi = analyse._berechne_rsi(preise)
    
    assert isinstance(rsi, pd.Series)
    assert not rsi.isna().all()
    assert (rsi >= 0).all() and (rsi <= 100).all()

def test_berechne_macd():
    analyse = MarktAnalyse("TEST")
    preise = pd.Series(np.linspace(100, 200, 100))
    macd = analyse._berechne_macd(preise)
    
    assert isinstance(macd, pd.Series)
    assert not macd.isna().all()

def test_erstelle_bericht(mock_ticker):
    analyse = MarktAnalyse("TEST")
    bericht = analyse.erstelle_bericht()
    
    assert isinstance(bericht, str)
    assert "Marktanalyse für TEST" in bericht
    assert "Fundamentaldaten" in bericht
    assert "Technische Analyse" in bericht

if __name__ == "__main__":
    pytest.main([__file__])
""" 