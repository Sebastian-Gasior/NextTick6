"""
Tests für die Sentiment-Datenquellen
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ml4t_project.data.sentiment_data import SentimentDataProvider

class TestSentimentData:
    """Test-Suite für die Sentiment-Datenquelle"""
    
    @pytest.fixture
    def sentiment_provider(self):
        """Fixture für den Sentiment-Provider"""
        return SentimentDataProvider()
    
    @pytest.fixture
    def test_dates(self):
        """Fixture für Testdaten"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date
    
    def test_news_sentiment(self, sentiment_provider, test_dates):
        """Test für News-Sentiment"""
        start_date, end_date = test_dates
        
        # Hole Sentiment-Daten
        sentiment_df = sentiment_provider.get_news_sentiment('AAPL', start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert sentiment_df is not None
        assert isinstance(sentiment_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'sentiment_score' in sentiment_df.columns
        assert 'article_count' in sentiment_df.columns
        
        # Prüfe die Länge
        expected_days = (end_date - start_date).days + 1
        assert len(sentiment_df) == expected_days
        
        # Prüfe Wertebereich
        assert sentiment_df['sentiment_score'].min() >= -1
        assert sentiment_df['sentiment_score'].max() <= 1
        
    def test_social_sentiment(self, sentiment_provider, test_dates):
        """Test für Social Media Sentiment"""
        start_date, end_date = test_dates
        
        # Hole Sentiment-Daten
        sentiment_df = sentiment_provider.get_social_sentiment('MSFT', start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert sentiment_df is not None
        assert isinstance(sentiment_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'sentiment_score' in sentiment_df.columns
        assert 'tweet_volume' in sentiment_df.columns
        
        # Prüfe den Zeitindex (stündliche Daten)
        expected_hours = (end_date - start_date).total_seconds() / 3600 + 1
        assert abs(len(sentiment_df) - expected_hours) <= 1  # Berücksichtige Sommerzeit
        
        # Prüfe Wertebereich
        assert sentiment_df['sentiment_score'].min() >= -1
        assert sentiment_df['sentiment_score'].max() <= 1
        assert (sentiment_df['tweet_volume'] >= 0).all()
        
    def test_analyst_ratings(self, sentiment_provider):
        """Test für Analysten-Ratings"""
        # Hole Ratings
        ratings_df = sentiment_provider.get_analyst_ratings('GOOGL')
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert ratings_df is not None
        assert isinstance(ratings_df, pd.DataFrame)
        
        # Prüfe die Struktur
        assert 'analyst' in ratings_df.columns
        assert 'rating' in ratings_df.columns
        assert 'price_target' in ratings_df.columns
        
        # Prüfe Rating-Kategorien
        valid_ratings = ['Buy', 'Outperform', 'Hold', 'Underperform', 'Sell']
        assert all(rating in valid_ratings for rating in ratings_df['rating'])
        
        # Prüfe Analysten-Namen
        assert not ratings_df['analyst'].isna().any()
        assert len(ratings_df['analyst'].unique()) > 0
        
    def test_combined_sentiment(self, sentiment_provider, test_dates):
        """Test für kombiniertes Sentiment"""
        start_date, end_date = test_dates
        
        # Hole kombiniertes Sentiment
        combined_df = sentiment_provider.get_combined_sentiment('AMZN', start_date, end_date)
        
        # Prüfe, ob ein DataFrame zurückgegeben wurde
        assert combined_df is not None
        assert isinstance(combined_df, pd.DataFrame)
        
        # Prüfe die Struktur - mindestens eine Sentiment-Quelle und der kombinierte Score
        sentiment_columns = [col for col in combined_df.columns if 'sentiment' in col or 'rating' in col]
        assert len(sentiment_columns) >= 2
        assert 'combined_sentiment' in combined_df.columns
        
        # Prüfe die Länge (tägliche Daten)
        expected_days = (end_date - start_date).days + 1
        assert len(combined_df) == expected_days
        
        # Prüfe Wertebereich
        assert combined_df['combined_sentiment'].min() >= -1
        assert combined_df['combined_sentiment'].max() <= 1
        
    def test_different_symbols(self, sentiment_provider):
        """Test für verschiedene Symbole"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for symbol in symbols:
            # Hole kombiniertes Sentiment
            combined_df = sentiment_provider.get_combined_sentiment(symbol)
            
            # Prüfe, dass für jedes Symbol Daten generiert werden
            assert combined_df is not None
            assert isinstance(combined_df, pd.DataFrame)
            
            # Verschiedene Symbole sollten unterschiedliche Sentiment-Werte haben
            if symbol != symbols[0]:
                previous_df = sentiment_provider.get_combined_sentiment(symbols[0])
                assert not combined_df['combined_sentiment'].equals(previous_df['combined_sentiment'])
                
    def test_error_handling(self, sentiment_provider):
        """Test für Fehlerbehandlung"""
        # Test mit ungültigem Datum (Enddatum vor Startdatum)
        future_date = datetime.now() + timedelta(days=30)
        past_date = datetime.now() - timedelta(days=30)
        
        # Bei falscher Reihenfolge der Datumsangaben sollte trotzdem ein DataFrame zurückgegeben werden,
        # da der Provider das Datum korrigieren oder mit Standardwerten arbeiten sollte
        result = sentiment_provider.get_combined_sentiment('INVALID_SYMBOL', future_date, past_date)
        
        # Überprüfen, dass wir ein Ergebnis erhalten und es ein DataFrame ist (leeres oder nicht)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        
        # Alternativ können wir prüfen, ob bei ungültigen Symbolen
        # trotzdem synthetische Daten generiert werden
        assert not result.empty
        assert 'combined_sentiment' in result.columns 