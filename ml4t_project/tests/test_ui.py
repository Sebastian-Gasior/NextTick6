"""
UI-Tests für die Visualisierungskomponenten des ML4T-Systems
"""
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from ml4t_project.ui.dashboard import TradingDashboard

from ml4t_project.visual.plotter import plot_signals, plot_predictions, plot_performance
from ml4t_project.features.indicators import add_all_indicators

def generate_test_data(days: int = 100) -> pd.DataFrame:
    """Generiert Testdaten für UI-Tests"""
    dates = pd.date_range(start='2020-01-01', periods=days)
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Open': np.random.normal(100, 10, days),
        'High': np.random.normal(102, 10, days),
        'Low': np.random.normal(98, 10, days),
        'Close': np.random.normal(100, 10, days),
        'Volume': np.random.normal(1000000, 100000, days)
    }, index=dates)
    
    return add_all_indicators(df)

@pytest.mark.ui
def test_plot_signals_basic():
    """Test der grundlegenden Signalvisualisierung"""
    # Testdaten vorbereiten
    df = generate_test_data()
    signals = np.random.choice([-1, 0, 1], size=len(df))
    
    # Plot erstellen
    fig = plot_signals(df, signals)
    
    # Überprüfungen
    assert isinstance(fig, go.Figure), "Rückgabewert sollte eine Plotly-Figure sein"
    assert len(fig.data) >= 3, "Plot sollte mindestens 3 Traces haben (Preis, Kauf, Verkauf)"
    assert fig.layout.title is not None, "Plot sollte einen Titel haben"
    assert fig.layout.xaxis.title is not None, "X-Achse sollte beschriftet sein"
    assert fig.layout.yaxis.title is not None, "Y-Achse sollte beschriftet sein"

@pytest.mark.ui
def test_plot_signals_interactivity():
    """Test der interaktiven Elemente"""
    df = generate_test_data()
    signals = np.random.choice([-1, 0, 1], size=len(df))
    
    fig = plot_signals(df, signals)
    
    # Überprüfe Hover-Informationen
    for trace in fig.data:
        assert trace.hovertemplate is not None, "Alle Traces sollten Hover-Templates haben"
    
    # Überprüfe Zoom-Funktionalität
    assert fig.layout.xaxis.rangeslider is not None, "Plot sollte einen Rangeslider haben"

@pytest.mark.ui
def test_plot_predictions():
    """Test der Vorhersage-Visualisierung"""
    df = generate_test_data()
    predictions = df['Close'].values + np.random.normal(0, 2, len(df))
    
    fig = plot_predictions(df['Close'], predictions)
    
    # Überprüfungen
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2, "Plot sollte 2 Linien haben (Actual und Predicted)"
    
    # Überprüfe Legende
    legend_labels = [trace.name for trace in fig.data]
    assert "Actual" in legend_labels, "Plot sollte 'Actual' in der Legende haben"
    assert "Predicted" in legend_labels, "Plot sollte 'Predicted' in der Legende haben"

@pytest.mark.ui
def test_plot_performance():
    """Test der Performance-Visualisierung"""
    df = generate_test_data()
    portfolio_value = np.cumprod(1 + np.random.normal(0.001, 0.02, len(df)))
    benchmark_value = np.cumprod(1 + np.random.normal(0.0008, 0.015, len(df)))
    
    fig = plot_performance(df.index, portfolio_value, benchmark_value)
    
    # Überprüfungen
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2, "Plot sollte 2 Linien haben (Portfolio und Benchmark)"
    
    # Überprüfe Y-Achsen-Format
    assert fig.layout.yaxis.tickformat == '.2%', "Y-Achse sollte Prozentformat haben"

@pytest.mark.ui
def test_responsive_layout():
    """Test des responsiven Layouts"""
    df = generate_test_data()
    signals = np.random.choice([-1, 0, 1], size=len(df))
    
    fig = plot_signals(df, signals)
    
    # Überprüfe Responsive-Einstellungen
    assert fig.layout.autosize is True, "Plot sollte responsive sein"
    assert fig.layout.margin is not None, "Plot sollte Margins haben"

@pytest.mark.ui
def test_error_handling():
    """Test der Fehlerbehandlung in der Visualisierung"""
    df = generate_test_data()
    
    # Test mit leeren Daten
    with pytest.raises(ValueError):
        plot_signals(pd.DataFrame(), np.array([]))
    
    # Test mit nicht übereinstimmenden Längen
    with pytest.raises(ValueError):
        plot_signals(df, np.array([1, 2, 3]))
    
    # Test mit ungültigen Signalwerten
    with pytest.raises(ValueError):
        plot_signals(df, np.array([2] * len(df)))

@pytest.mark.ui
def test_theme_consistency():
    """Test der Theme-Konsistenz"""
    df = generate_test_data()
    signals = np.random.choice([-1, 0, 1], size=len(df))
    predictions = df['Close'].values + np.random.normal(0, 2, len(df))
    portfolio_value = np.cumprod(1 + np.random.normal(0.001, 0.02, len(df)))
    benchmark_value = np.cumprod(1 + np.random.normal(0.0008, 0.015, len(df)))
    
    # Erstelle alle Plot-Typen
    fig_signals = plot_signals(df, signals)
    fig_predictions = plot_predictions(df['Close'], predictions)
    fig_performance = plot_performance(df.index, portfolio_value, benchmark_value)
    
    # Überprüfe Theme-Konsistenz
    plots = [fig_signals, fig_predictions, fig_performance]
    first_plot_template = plots[0].layout.template
    
    for fig in plots[1:]:
        assert fig.layout.template == first_plot_template, "Alle Plots sollten das gleiche Theme verwenden"
        
    # Überprüfe Farbschema-Konsistenz
    for fig in plots:
        assert fig.layout.plot_bgcolor == 'white', "Alle Plots sollten den gleichen Hintergrund haben"
        assert fig.layout.paper_bgcolor == 'white', "Alle Plots sollten den gleichen Papierhintergrund haben"

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach jedem Test."""
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class TestTradingDashboard:
    def setup_method(self):
        self.dashboard = TradingDashboard(
            update_interval=timedelta(seconds=1),
            max_data_points=1000
        )
        
    def test_initialization(self):
        assert self.dashboard.is_initialized
        assert self.dashboard.update_interval == timedelta(seconds=1)
        assert self.dashboard.max_data_points == 1000
        
    def test_data_visualization(self):
        data = torch.randn(100, 5)
        success = self.dashboard.update_plot(data)
        assert success
        
    def test_performance_metrics(self):
        metrics = {
            'accuracy': 0.85,
            'profit': 1000.0,
            'trades': 50
        }
        success = self.dashboard.update_metrics(metrics)
        assert success
        
    def test_alerts(self):
        alert = {
            'level': 'warning',
            'message': 'Test Alert',
            'timestamp': datetime.now()
        }
        success = self.dashboard.add_alert(alert)
        assert success
        
    def test_error_handling(self):
        with pytest.raises(ValueError):
            self.dashboard.update_plot(None)
            
    def test_performance_metrics(self):
        metrics = self.dashboard.get_performance_metrics()
        assert 'update_rate' in metrics
        assert 'memory_usage' in metrics
        assert 'latency' in metrics
        
    def test_cleanup(self):
        self.dashboard.cleanup()
        assert not self.dashboard.is_initialized

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nicht verfügbar")
    def test_gpu_accelerated_plotting(self):
        """Test GPU-beschleunigte Ploterstellung."""
        dashboard = TradingDashboard(use_gpu=True)
        data = torch.randn(1000, 4).cuda()  # Simulierte Handelsdaten auf GPU
        
        # Test Plot-Erstellung mit GPU-Daten
        plot = dashboard.create_plot(data)
        assert plot is not None
        assert dashboard.is_gpu_accelerated
        
        # Test Performance-Metriken
        metrics = dashboard.get_gpu_metrics()
        assert 'gpu_render_time' in metrics
        assert metrics['gpu_render_time'] > 0 