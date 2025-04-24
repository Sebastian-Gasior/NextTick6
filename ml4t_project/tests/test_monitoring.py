"""
Tests für das Monitoring-Dashboard.
"""

import pytest
from dash.testing.application_runners import import_app
from dash.testing.composite import DashComposite
import time
from ..monitoring.dashboard import MonitoringDashboard
import torch
from ml4t_project.monitoring.system_monitor import SystemMonitor
from datetime import datetime, timedelta

@pytest.fixture
def dashboard():
    """Fixture für MonitoringDashboard"""
    return MonitoringDashboard(
        log_dir="tests/logs",
        history_size=10,
        update_interval=100
    )

def test_dashboard_initialization(dashboard):
    """Test Dashboard-Initialisierung"""
    assert dashboard.history_size == 10
    assert dashboard.update_interval == 100
    assert dashboard.app is not None
    assert len(dashboard.cpu_history) == 0
    assert len(dashboard.memory_history) == 0
    assert len(dashboard.gpu_history) == 0
    assert len(dashboard.throughput_history) == 0

def test_logger_setup(dashboard):
    """Test Logger-Setup"""
    assert dashboard.logger is not None
    assert dashboard.logger.name == 'MonitoringDashboard'
    assert len(dashboard.logger.handlers) > 0

@pytest.mark.integration
def test_dashboard_layout(dashboard):
    """Test Dashboard-Layout"""
    layout = dashboard.app.layout
    
    # Prüfe Hauptkomponenten
    assert 'system-metrics' in layout.children[1].children[1].id
    assert 'gpu-metrics' in layout.children[2].children[1].id
    assert 'throughput-metrics' in layout.children[3].children[1].id
    assert 'alerts-container' in layout.children[4].children[1].id

def test_metric_updates(dashboard):
    """Test Metrik-Updates"""
    # System-Metriken
    figure = dashboard.app.callback_map['system-metrics']['callback']()
    assert isinstance(figure, dict)
    assert 'data' in figure
    assert len(figure['data']) == 2  # CPU und Memory
    
    # GPU-Metriken
    figure = dashboard.app.callback_map['gpu-metrics']['callback']()
    assert isinstance(figure, dict)
    
    # Durchsatz-Metriken
    figure = dashboard.app.callback_map['throughput-metrics']['callback']()
    assert isinstance(figure, dict)
    assert 'data' in figure
    assert len(figure['data']) == 1

def test_alert_system(dashboard):
    """Test Alert-System"""
    alerts = dashboard.app.callback_map['alerts-container']['callback']()
    assert isinstance(alerts, list)
    assert len(alerts) > 0

@pytest.mark.performance
def test_update_performance(dashboard):
    """Test Update-Performance"""
    import time
    
    start_time = time.time()
    
    # Führe 100 Updates durch
    for _ in range(100):
        dashboard.app.callback_map['system-metrics']['callback']()
        dashboard.app.callback_map['gpu-metrics']['callback']()
        dashboard.app.callback_map['throughput-metrics']['callback']()
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Updates sollten schnell sein (< 1s für 100 Updates)
    assert execution_time < 1.0

@pytest.mark.integration
def test_data_history(dashboard):
    """Test Daten-Historie"""
    # Fülle Historie
    for _ in range(15):  # Mehr als history_size
        dashboard.app.callback_map['system-metrics']['callback']()
    
    # Prüfe Größenbeschränkung
    assert len(dashboard.cpu_history) == dashboard.history_size
    assert len(dashboard.memory_history) == dashboard.history_size

def test_error_handling(dashboard):
    """Test Fehlerbehandlung"""
    # Simuliere GPU-Fehler
    import GPUtil
    GPUtil.getGPUs = lambda: []  # Keine GPUs verfügbar
    
    # Sollte nicht abstürzen
    figure = dashboard.app.callback_map['gpu-metrics']['callback']()
    assert isinstance(figure, dict)
    
    # Simuliere Systemfehler
    import psutil
    original_cpu_percent = psutil.cpu_percent
    psutil.cpu_percent = lambda: float('nan')
    
    # Sollte nicht abstürzen
    figure = dashboard.app.callback_map['system-metrics']['callback']()
    assert isinstance(figure, dict)
    
    # Cleanup
    psutil.cpu_percent = original_cpu_percent

@pytest.mark.integration
def test_full_dashboard_cycle(dashboard):
    """Test vollständiger Dashboard-Zyklus"""
    # Simuliere mehrere Update-Zyklen
    for _ in range(5):
        # Update alle Metriken
        dashboard.app.callback_map['system-metrics']['callback']()
        dashboard.app.callback_map['gpu-metrics']['callback']()
        dashboard.app.callback_map['throughput-metrics']['callback']()
        dashboard.app.callback_map['alerts-container']['callback']()
        
        # Kurze Pause zwischen Updates
        time.sleep(0.1)
    
    # Prüfe Datensammlung
    assert len(dashboard.cpu_history) > 0
    assert len(dashboard.memory_history) > 0
    
    # Prüfe Alert-System
    alerts = dashboard.app.callback_map['alerts-container']['callback']()
    assert isinstance(alerts, list)

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach Tests"""
    yield
    import shutil
    try:
        shutil.rmtree("tests/logs")
    except:
        pass

class TestSystemMonitor:
    def setup_method(self):
        self.monitor = SystemMonitor(
            sampling_interval=timedelta(seconds=1),
            alert_thresholds={
                'cpu_usage': 90,
                'memory_usage': 85,
                'gpu_usage': 95,
                'disk_usage': 90
            }
        )
        
    def test_initialization(self):
        assert self.monitor.is_initialized
        assert self.monitor.sampling_interval == timedelta(seconds=1)
        assert self.monitor.alert_thresholds['cpu_usage'] == 90
        
    def test_metric_collection(self):
        metrics = self.monitor.collect_metrics()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'gpu_usage' in metrics
        assert 'disk_usage' in metrics
        
    def test_alert_generation(self):
        metrics = {
            'cpu_usage': 95,
            'memory_usage': 90,
            'gpu_usage': 98,
            'disk_usage': 95
        }
        alerts = self.monitor.check_alerts(metrics)
        assert len(alerts) > 0
        assert all('level' in alert for alert in alerts)
        
    def test_historical_data(self):
        history = self.monitor.get_historical_data(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        assert 'metrics' in history
        assert 'alerts' in history
        
    def test_performance_trends(self):
        trends = self.monitor.analyze_trends()
        assert 'cpu_trend' in trends
        assert 'memory_trend' in trends
        assert 'gpu_trend' in trends
        assert 'disk_trend' in trends
        
    def test_error_handling(self):
        with pytest.raises(ValueError):
            self.monitor.collect_metrics(invalid_param=True)
            
    def test_report_generation(self):
        report = self.monitor.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'alerts' in report
        assert 'recommendations' in report
        
    def test_cleanup(self):
        self.monitor.cleanup()
        assert not self.monitor.is_initialized 