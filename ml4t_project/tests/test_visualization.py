"""
Tests für die Visualisierungskomponenten
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml4t_project.visualization.show_chart import get_lstm_predictions, main
import plotly.graph_objects as go

@pytest.fixture
def sample_data():
    """Erstellt Beispieldaten für die Tests."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    data = {
        'Open': np.random.normal(100, 5, len(dates)),
        'High': np.random.normal(105, 5, len(dates)),
        'Low': np.random.normal(95, 5, len(dates)),
        'Close': np.random.normal(100, 5, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_lstm_predictions(sample_data):
    """Testet die LSTM-Vorhersagegenerierung."""
    predictions = get_lstm_predictions(sample_data)
    
    # Überprüfe die Vorhersagen
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) > 0
    assert predictions.shape[1] == 1  # Eine Spalte für die Vorhersage
    
    # Überprüfe die Wertebereiche
    assert np.all(np.isfinite(predictions))
    assert np.min(predictions) >= sample_data['Low'].min() * 0.9
    assert np.max(predictions) <= sample_data['High'].max() * 1.1

def test_visualization_integration(sample_data):
    """Testet die Integration der LSTM-Vorhersagen in die Visualisierung."""
    # Erstelle eine Figure mit den Beispieldaten
    fig = go.Figure()
    
    # Füge die Daten hinzu
    fig.add_trace(go.Candlestick(
        x=sample_data.index,
        open=sample_data['Open'],
        high=sample_data['High'],
        low=sample_data['Low'],
        close=sample_data['Close'],
        name='TEST'
    ))
    
    # Füge die LSTM-Vorhersagen hinzu
    predictions = get_lstm_predictions(sample_data)
    sample_data['LSTM_Prediction'] = np.nan
    sample_data.iloc[60:len(predictions)+60, sample_data.columns.get_loc('LSTM_Prediction')] = predictions.flatten()
    
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['LSTM_Prediction'],
        name='LSTM-Vorhersage',
        line=dict(color='#e91e63', width=2),
        opacity=0.7
    ))
    
    # Überprüfe die Figure
    assert fig is not None
    assert len(fig.data) == 2  # Candlestick und LSTM-Vorhersage
    
    # Überprüfe die LSTM-Vorhersage-Spur
    lstm_trace = next((trace for trace in fig.data if trace.name == 'LSTM-Vorhersage'), None)
    assert lstm_trace is not None
    assert len(lstm_trace.x) > 0
    assert len(lstm_trace.y) > 0
    
    # Überprüfe die Visualisierungseigenschaften
    assert lstm_trace.line.color == '#e91e63'
    assert lstm_trace.line.width == 2
    assert lstm_trace.opacity == 0.7

def test_error_handling():
    """Testet die Fehlerbehandlung bei ungültigen Eingaben."""
    # Test mit leeren Daten
    with pytest.raises(ValueError):
        get_lstm_predictions(pd.DataFrame())
    
    # Test mit zu wenigen Datenpunkten
    with pytest.raises(ValueError):
        get_lstm_predictions(pd.DataFrame({'Close': [100] * 10}))
    
    # Test mit ungültigen Symbolen
    with pytest.raises(Exception):
        main(symbol="INVALID_SYMBOL", start_date="2023-01-01", end_date="2023-12-31") 