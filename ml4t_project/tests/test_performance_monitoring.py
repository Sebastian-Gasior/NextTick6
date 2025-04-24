import pytest
import torch
from ml4t_project.monitoring.performance_monitor import PerformanceMonitor
from datetime import datetime, timedelta

class TestPerformanceMonitor:
    def setup_method(self):
        self.monitor = PerformanceMonitor(
            sampling_interval=timedelta(seconds=1),
            alert_thresholds={
                'cpu_usage': 90,
                'memory_usage': 85,
                'gpu_usage': 95,
                'latency': 1000
            }
        )
        
    def test_metric_collection(self):
        # Teste Metrikensammlung
        metrics = self.monitor.collect_metrics()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'gpu_usage' in metrics
        assert 'latency' in metrics
        
    def test_alert_generation(self):
        # Teste Alert-Generierung
        metrics = {
            'cpu_usage': 95,
            'memory_usage': 90,
            'gpu_usage': 98,
            'latency': 1200
        }
        alerts = self.monitor.check_alerts(metrics)
        assert len(alerts) > 0
        assert all('level' in alert for alert in alerts)
        
    def test_historical_data(self):
        # Teste historische Daten
        history = self.monitor.get_historical_data(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        assert 'metrics' in history
        assert 'alerts' in history
        
    def test_performance_trends(self):
        # Teste Performance-Trends
        trends = self.monitor.analyze_trends()
        assert 'cpu_trend' in trends
        assert 'memory_trend' in trends
        assert 'gpu_trend' in trends
        
    def test_error_handling(self):
        # Teste Fehlerbehandlung
        with pytest.raises(ValueError):
            self.monitor.collect_metrics(invalid_param=True)
            
    def test_report_generation(self):
        # Teste Berichtgenerierung
        report = self.monitor.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'alerts' in report
        assert 'recommendations' in report
        
    def test_cleanup(self):
        # Teste Bereinigung
        self.monitor.cleanup()
        assert not self.monitor.is_active 