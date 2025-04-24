import pytest
from ml4t_project.data.market_data import MarketData
from ml4t_project.analysis.market_analyzer import MarketAnalyzer
import pandas as pd
from datetime import datetime
import pytz

@pytest.fixture
def market_data():
    return MarketData()

@pytest.fixture
def market_analyzer():
    return MarketAnalyzer()

def test_yahoo_data_download(market_data):
    """Testet den Download von Yahoo Finance Daten"""
    # Test für AAPL Aktie
    df = market_data.get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Überprüfe grundlegende Eigenschaften
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'Date' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns
    
    # Überprüfe Datentypen
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    assert pd.api.types.is_numeric_dtype(df['Close'])
    assert pd.api.types.is_numeric_dtype(df['Volume'])
    
    # Überprüfe Zeitraum (mit Zeitzonen)
    start_date = df['Date'].min().tz_localize(None)
    end_date = df['Date'].max().tz_localize(None)
    assert start_date >= pd.Timestamp('2023-01-01')
    assert end_date <= pd.Timestamp('2024-01-01')

def test_market_analysis(market_data, market_analyzer):
    """Testet die Marktanalyse mit Yahoo Finance Daten"""
    # Lade Testdaten
    df = market_data.get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Führe Analyse durch
    analysis = market_analyzer.analyze_stock(df, 'AAPL')
    
    # Überprüfe Analyseergebnisse
    assert isinstance(analysis, str)
    assert 'Marktanalyse für AAPL' in analysis
    assert 'Grundlegende Statistiken' in analysis
    assert 'Handelssignale' in analysis
    assert 'Performance der Trades' in analysis

def test_technical_indicators(market_data):
    """Testet die Berechnung technischer Indikatoren"""
    df = market_data.get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Überprüfe technische Indikatoren
    assert 'SMA20' in df.columns
    assert 'SMA50' in df.columns
    assert 'RSI' in df.columns
    
    # Überprüfe Berechnungen
    assert not df['SMA20'].isna().all()
    assert not df['SMA50'].isna().all()
    assert not df['RSI'].isna().all()
    
    # Überprüfe Wertebereiche
    assert df['RSI'].min() >= 0
    assert df['RSI'].max() <= 100

def test_data_caching(market_data):
    """Testet das Caching der Daten"""
    # Erster Download
    df1 = market_data.get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Zweiter Download (sollte aus Cache kommen)
    df2 = market_data.get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Überprüfe Cache-Funktionalität
    assert df1.equals(df2) 