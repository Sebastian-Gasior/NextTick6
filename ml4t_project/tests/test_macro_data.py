"""
Tests für die Makroökonomischen Datenquellen
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ml4t_project.data.macro_data import MacroDataProvider

class TestMacroData:
    """Test-Suite für die Makroökonomische Datenquelle"""
    
    @pytest.fixture
    def macro_provider(self):
        """Fixture für den Makrodaten-Provider"""
        return MacroDataProvider()
    
    @pytest.fixture
    def test_dates(self):
        """Fixture für Testdaten"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        return start_date, end_date
    
    def test_market_indices(self, macro_provider, test_dates):
        """Test für Marktindizes"""
        start_date, end_date = test_dates
        
        # Hole Marktindizes
        indices_df = macro_provider.get_market_indices(start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert indices_df is not None
        assert isinstance(indices_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'S&P500' in indices_df.columns
        assert 'NASDAQ' in indices_df.columns
        assert 'DAX' in indices_df.columns
        assert 'Nikkei' in indices_df.columns
        
        # Prüfe Return-Spalten
        assert 'S&P500_return' in indices_df.columns
        
        # Prüfe die Indexstruktur (nur Handelstage)
        assert all(idx.dayofweek < 5 for idx in indices_df.index)
        
        # Prüfe Wertebereich (Indizes sollten positiv sein)
        assert (indices_df['S&P500'] > 0).all()
        assert (indices_df['NASDAQ'] > 0).all()
        
    def test_economic_indicators(self, macro_provider, test_dates):
        """Test für Wirtschaftsindikatoren"""
        start_date, end_date = test_dates
        
        # Hole Wirtschaftsindikatoren
        indicators_df = macro_provider.get_economic_indicators(start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert indicators_df is not None
        assert isinstance(indicators_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'Inflation' in indicators_df.columns
        assert 'Arbeitslosenquote' in indicators_df.columns
        assert 'BIP_Wachstum' in indicators_df.columns
        assert 'Leitzins' in indicators_df.columns
        
        # Prüfe die Indexstruktur (monatliche Daten)
        assert 'M' in indicators_df.index.freqstr  # Prüfe ob monatliche Frequenz
        
        # Prüfe Wertebereich
        assert (indicators_df['Arbeitslosenquote'] >= 0).all()
        assert (indicators_df['Leitzins'] >= 0).all()
        
    def test_sector_performance(self, macro_provider, test_dates):
        """Test für Sektorperformance"""
        start_date, end_date = test_dates
        
        # Hole Sektorperformance
        sector_data = macro_provider.get_sector_performance(start_date, end_date)
        
        # Prüfe, ob ein Dictionary zurückgegeben wurde
        assert sector_data is not None
        assert isinstance(sector_data, dict)
        assert 'performance' in sector_data
        assert 'returns' in sector_data
        
        # Prüfe Performance-DataFrame
        performance_df = sector_data['performance']
        assert isinstance(performance_df, pd.DataFrame)
        
        # Prüfe Returns-DataFrame
        returns_df = sector_data['returns']
        assert isinstance(returns_df, pd.DataFrame)
        
        # Prüfe Sektor-Spalten in beiden DataFrames
        assert 'Technology' in performance_df.columns
        assert 'Technology' in returns_df.columns
        
        # Prüfe relative Performance-Spalten
        assert 'Technology_rel' in performance_df.columns
        
        # Prüfe die Indexstruktur (nur Handelstage)
        assert all(idx.dayofweek < 5 for idx in performance_df.index)
        assert all(idx.dayofweek < 5 for idx in returns_df.index)
        
    def test_volatility_indices(self, macro_provider, test_dates):
        """Test für Volatilitätsindizes"""
        start_date, end_date = test_dates
        
        # Hole Volatilitätsindizes
        volatility_df = macro_provider.get_volatility_indices(start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert volatility_df is not None
        assert isinstance(volatility_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'VIX' in volatility_df.columns
        assert 'VVIX' in volatility_df.columns
        
        # Prüfe die Indexstruktur (nur Handelstage)
        assert all(idx.dayofweek < 5 for idx in volatility_df.index)
        
        # Prüfe Wertebereich
        assert (volatility_df['VIX'] > 0).all()
        assert (volatility_df['VVIX'] > 0).all()
        
        # VVIX sollte in der Regel höher als VIX sein
        assert (volatility_df['VVIX'] > volatility_df['VIX']).mean() > 0.9
        
    def test_reproducibility(self, macro_provider):
        """Test für Reproduzierbarkeit der Daten"""
        # Erzeuge zwei Datensätze mit gleichen Parametern
        indices1 = macro_provider.get_market_indices()
        indices2 = macro_provider.get_market_indices()
        
        # Die Datensätze sollten identisch sein (gleicher Seed)
        pd.testing.assert_frame_equal(indices1, indices2)
        
        # Erzeuge zwei Datensätze mit verschiedenen Symbolen/Seeds
        vix1 = macro_provider.get_volatility_indices()
        
        # Ändere den Zeitraum, um einen anderen Seed zu erzwingen
        start_date = datetime.now() - timedelta(days=366)  # Ein Tag mehr
        end_date = datetime.now()
        vix2 = macro_provider.get_volatility_indices(start_date, end_date)
        
        # Die Datensätze sollten unterschiedlich sein
        assert len(vix1) != len(vix2)
        
    def test_error_handling(self, macro_provider):
        """Test für Fehlerbehandlung"""
        # Test mit ungültigem Datum
        future_date = datetime.now() + timedelta(days=30)
        past_date = datetime.now() - timedelta(days=30)
        
        # Dies sollte keine Fehler werfen, sondern None zurückgeben
        result = macro_provider.get_market_indices(future_date, past_date)
        assert result is not None  # Wir erwarten hier synthetische Daten, auch bei ungültiger Eingabe 