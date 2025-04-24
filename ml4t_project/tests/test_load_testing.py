import pytest
import torch
from ml4t_project.testing.load_tester import LoadTester
from datetime import datetime, timedelta

class TestLoadTester:
    def setup_method(self):
        self.tester = LoadTester(
            target_throughput=1000,
            duration=timedelta(seconds=10),
            ramp_up_time=timedelta(seconds=2)
        )
        
    def test_load_generation(self):
        # Teste Lastgenerierung
        load_config = {
            'batch_size': 32,
            'model_size': 'medium',
            'data_size': 1000
        }
        results = self.tester.generate_load(load_config)
        assert 'success_rate' in results
        assert 'error_rate' in results
        assert 'latency' in results
        
    def test_ramp_up_behavior(self):
        # Teste Ramp-up-Verhalten
        metrics = self.tester.test_ramp_up()
        assert 'start_time' in metrics
        assert 'end_time' in metrics
        assert 'peak_load' in metrics
        
    def test_error_handling(self):
        # Teste Fehlerbehandlung
        with pytest.raises(ValueError):
            self.tester.generate_load({'invalid': 'config'})
            
    def test_performance_metrics(self):
        # Teste Performance-Metriken
        metrics = self.tester.get_performance_metrics()
        assert 'throughput' in metrics
        assert 'latency' in metrics
        assert 'error_rate' in metrics
        assert 'resource_usage' in metrics
        
    def test_resource_monitoring(self):
        # Teste Ressourcenüberwachung
        resources = self.tester.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'gpu_usage' in resources
        assert 'network_usage' in resources
        
    def test_high_load_scenario(self):
        # Teste Hochlast-Szenario
        results = self.tester.test_high_load()
        assert 'success_rate' in results
        assert 'error_rate' in results
        assert 'peak_latency' in results
        
    def test_report_generation(self):
        # Teste Berichtgenerierung
        report = self.tester.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'recommendations' in report 