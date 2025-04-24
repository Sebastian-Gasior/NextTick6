"""
Tests für die Orderflow-Datenquellen
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml4t_project.data.orderflow_data import OrderFlowDataProvider

class TestOrderFlowData:
    """Test-Suite für die OrderFlow-Datenquelle"""
    
    @pytest.fixture
    def orderflow_provider(self):
        """Fixture für den OrderFlow-Provider"""
        return OrderFlowDataProvider()
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Fixture für OHLCV-Testdaten"""
        # Erzeuge Testdaten für 10 Tage
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=10, freq='D')
        
        # Generiere OHLCV-Daten
        data = {
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(105, 115, 10),
            'Low': np.linspace(95, 105, 10),
            'Close': np.linspace(102, 112, 10),
            'Volume': np.random.randint(1000, 5000, 10)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_volume_profile(self, orderflow_provider, sample_ohlcv_data):
        """Test für Volumen-Profil"""
        # Erzeuge Volumen-Profil
        profile = orderflow_provider.get_volume_profile(sample_ohlcv_data)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert profile is not None
        assert isinstance(profile, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'Price' in profile.columns
        assert 'TotalVolume' in profile.columns
        assert 'BuyVolume' in profile.columns
        assert 'SellVolume' in profile.columns
        assert 'BuySellRatio' in profile.columns
        
        # Prüfe die Anzahl der Bins (Standard: 20)
        assert len(profile) == 20
        
        # Prüfe Sortierung (absteigend nach Preis)
        assert profile['Price'].is_monotonic_decreasing
        
        # Prüfe Wertebereiche
        assert (profile['TotalVolume'] >= 0).all()
        assert (profile['BuyVolume'] >= 0).all()
        assert (profile['SellVolume'] >= 0).all()
        
    def test_vwap_levels(self, orderflow_provider, sample_ohlcv_data):
        """Test für VWAP-Levels"""
        # Berechne VWAP-Levels
        vwap_df = orderflow_provider.get_vwap_levels(sample_ohlcv_data)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert vwap_df is not None
        assert isinstance(vwap_df, pd.DataFrame)
        
        # Prüfe die Struktur (Standard-Perioden: [1, 5, 10, 20])
        assert 'VWAP_1' in vwap_df.columns
        assert 'VWAP_5' in vwap_df.columns
        assert 'VWAP_10' in vwap_df.columns
        
        # Prüfe, dass Hilfsspalten entfernt wurden
        assert 'TypicalPrice' not in vwap_df.columns
        assert 'VolumePrice' not in vwap_df.columns
        
        # Prüfe Werte
        # VWAP sollte innerhalb des Bereichs [Low, High] liegen
        non_na_mask = ~np.isnan(vwap_df['VWAP_1'])
        if non_na_mask.any():
            min_price = sample_ohlcv_data.loc[non_na_mask, 'Low'].min()
            max_price = sample_ohlcv_data.loc[non_na_mask, 'High'].max()
            assert vwap_df.loc[non_na_mask, 'VWAP_1'].min() >= min_price * 0.9  # Etwas Toleranz
            assert vwap_df.loc[non_na_mask, 'VWAP_1'].max() <= max_price * 1.1  # Etwas Toleranz
        
    def test_volume_indicators(self, orderflow_provider, sample_ohlcv_data):
        """Test für Volumenindikatoren"""
        # Berechne Volumenindikatoren
        vol_df = orderflow_provider.get_volume_indicators(sample_ohlcv_data)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert vol_df is not None
        assert isinstance(vol_df, pd.DataFrame)
        
        # Prüfe die berechneten Indikatoren
        assert 'Volume_MA_10' in vol_df.columns
        assert 'Volume_MA_20' in vol_df.columns
        assert 'Volume_MA_50' in vol_df.columns
        assert 'Relative_Volume' in vol_df.columns
        assert 'OBV' in vol_df.columns
        assert 'CMF' in vol_df.columns
        assert 'VPT' in vol_df.columns
        
        # Prüfe, dass Hilfsspalten entfernt wurden
        assert 'MFM' not in vol_df.columns
        assert 'MFV' not in vol_df.columns
        assert 'Price_ROC' not in vol_df.columns
        
        # Prüfe OBV-Berechnung
        # OBV sollte mit dem ursprünglichen Volumen beginnen
        assert vol_df['OBV'].iloc[0] == sample_ohlcv_data['Volume'].iloc[0]
        
    def test_synthetic_orderbook(self, orderflow_provider, sample_ohlcv_data):
        """Test für synthetisches Orderbuch"""
        # Generiere synthetisches Orderbuch
        orderbooks = orderflow_provider.generate_synthetic_orderbook(sample_ohlcv_data)
        
        # Prüfe, ob ein Dictionary zurückgegeben wurde
        assert orderbooks is not None
        assert isinstance(orderbooks, dict)
        
        # Prüfe, ob für jeden Zeitpunkt ein Orderbuch erstellt wurde
        assert len(orderbooks) == len(sample_ohlcv_data)
        
        # Prüfe ein Beispiel-Orderbuch
        timestamp = sample_ohlcv_data.index[0]
        orderbook = orderbooks[timestamp]
        
        assert 'BidPrice' in orderbook.columns
        assert 'BidVolume' in orderbook.columns
        assert 'AskPrice' in orderbook.columns
        assert 'AskVolume' in orderbook.columns
        assert 'Spread' in orderbook.columns
        assert 'MidPrice' in orderbook.columns
        
        # Prüfe Preisniveaus (Standard: 10)
        assert len(orderbook) == 10
        
        # Prüfe, dass Bid < Ask
        assert orderbook['BidPrice'].max() < orderbook['AskPrice'].min()
        
        # Prüfe, dass Volumina positiv sind
        assert (orderbook['BidVolume'] > 0).all()
        assert (orderbook['AskVolume'] > 0).all()
        
    def test_orderflow_metrics(self, orderflow_provider, sample_ohlcv_data):
        """Test für Order Flow Metriken"""
        # Berechne Order Flow Metriken
        metrics_df = orderflow_provider.get_orderflow_metrics(sample_ohlcv_data)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert metrics_df is not None
        assert isinstance(metrics_df, pd.DataFrame)
        
        # Prüfe die berechneten Metriken
        assert 'OrderImbalance' in metrics_df.columns
        assert 'Spread' in metrics_df.columns
        assert 'MarketDepth' in metrics_df.columns
        assert 'VolumeDelta' in metrics_df.columns
        assert 'CumulativeDelta' in metrics_df.columns
        
        # Prüfe OrderImbalance Bereich
        non_na_mask = ~np.isnan(metrics_df['OrderImbalance'])
        if non_na_mask.any():
            assert metrics_df.loc[non_na_mask, 'OrderImbalance'].min() >= -1
            assert metrics_df.loc[non_na_mask, 'OrderImbalance'].max() <= 1
        
        # Prüfe VolumeDelta
        for i, row in sample_ohlcv_data.iterrows():
            if row['Close'] >= row['Open']:
                assert metrics_df.loc[i, 'VolumeDelta'] == row['Volume']
            else:
                assert metrics_df.loc[i, 'VolumeDelta'] == -row['Volume']
                
    def test_empty_data(self, orderflow_provider):
        """Test mit leeren Daten"""
        # Erstelle leeren DataFrame
        empty_df = pd.DataFrame()
        
        # Prüfe, dass die Funktionen keine Fehler werfen
        profile = orderflow_provider.get_volume_profile(empty_df)
        assert isinstance(profile, pd.DataFrame)
        assert profile.empty
        
        orderbooks = orderflow_provider.generate_synthetic_orderbook(empty_df)
        assert isinstance(orderbooks, dict)
        assert len(orderbooks) == 0
        
        metrics = orderflow_provider.get_orderflow_metrics(empty_df)
        assert isinstance(metrics, pd.DataFrame)
        assert metrics.empty 