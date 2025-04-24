"""
Last-Tests für die Datenpipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import threading
import time
from ..data.data_migration import HybridDataPipeline
import torch
from ml4t_project.testing.load_stress import LoadStressTester

@pytest.fixture
def pipeline():
    """Fixture für HybridDataPipeline"""
    return HybridDataPipeline("tests/test_config.yaml")

@pytest.fixture
def large_symbol_list():
    """Fixture für große Symbol-Liste"""
    return [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'PFE',
        'CSCO', 'VZ', 'CVX', 'ADBE', 'CRM', 'ABT', 'TMO', 'COST', 'ACN',
        'DHR', 'MRK', 'AVGO', 'CMCSA', 'PEP', 'WFC', 'T', 'NKE'
    ]

@pytest.mark.stress
def test_concurrent_data_loading(pipeline, large_symbol_list):
    """Test parallele Datenverarbeitung"""
    def load_data(symbol):
        try:
            data = pipeline._get_real_data(1000, [symbol])
            assert len(data) > 0
            assert not data.empty
            return True
        except Exception:
            return False
    
    # Starte Threads für parallele Verarbeitung
    threads = []
    results = []
    
    start_time = time.time()
    
    for symbol in large_symbol_list:
        thread = threading.Thread(target=lambda: results.append(load_data(symbol)))
        threads.append(thread)
        thread.start()
    
    # Warte auf Abschluss
    for thread in threads:
        thread.join()
    
    execution_time = time.time() - start_time
    
    # Prüfe Ergebnisse
    success_rate = sum(results) / len(results)
    assert success_rate > 0.95  # 95% Erfolgsrate
    assert execution_time < 30.0  # Max 30 Sekunden

@pytest.mark.stress
def test_memory_management(pipeline):
    """Test Speichermanagement"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Lade große Datenmenge
    data = pipeline.get_data(10000)
    
    # Prüfe Speicherverbrauch
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    assert memory_increase < 2048  # Max 2GB Zunahme
    
    # Cleanup
    del data
    import gc
    gc.collect()
    
    # Prüfe Speicherbereinigung
    final_memory = process.memory_info().rss / 1024 / 1024
    assert (final_memory - initial_memory) < 100  # Max 100MB Differenz

@pytest.mark.stress
def test_cache_performance(pipeline):
    """Test Cache-Performance"""
    # Erste Ladung
    start_time = time.time()
    data1 = pipeline.get_data(1000)
    first_load_time = time.time() - start_time
    
    # Zweite Ladung (sollte gecacht sein)
    start_time = time.time()
    data2 = pipeline.get_data(1000)
    cached_load_time = time.time() - start_time
    
    # Cache sollte schneller sein
    assert cached_load_time < first_load_time * 0.5
    
    # Prüfe Datenkonsistenz
    pd.testing.assert_frame_equal(data1, data2)

@pytest.mark.stress
def test_long_running_stability(pipeline):
    """Test Langzeit-Stabilität"""
    start_time = time.time()
    end_time = start_time + 120  # 2 Minuten Test
    
    success_count = 0
    total_attempts = 0
    
    while time.time() < end_time:
        try:
            data = pipeline.get_data(1000)
            assert len(data) == 1000
            assert pipeline.validator.validate_structure(data)
            success_count += 1
        except Exception:
            pass
        total_attempts += 1
        time.sleep(1)  # Pause zwischen Anfragen
    
    success_rate = success_count / total_attempts
    assert success_rate > 0.95  # 95% Erfolgsrate

@pytest.mark.stress
def test_data_consistency_under_load(pipeline):
    """Test Datenkonsistenz unter Last"""
    # Sammle Statistiken über mehrere Ladungen
    stats_list = []
    
    for _ in range(10):
        data = pipeline.get_data(1000)
        stats = data.describe()
        stats_list.append(stats)
        
        # Prüfe Datenqualität
        assert pipeline.validator.validate_structure(data)
        is_valid, metrics = pipeline.validator.validate_data_quality(data)
        assert is_valid
    
    # Vergleiche Statistiken
    base_stats = stats_list[0]
    for stats in stats_list[1:]:
        pd.testing.assert_frame_equal(
            base_stats,
            stats,
            rtol=0.1  # 10% Toleranz
        )

@pytest.mark.stress
def test_error_recovery(pipeline):
    """Test Fehlerbehandlung unter Last"""
    def simulate_error():
        """Simuliert einen Fehler bei der Datenabfrage"""
        raise ConnectionError("Simulierter Netzwerkfehler")
    
    # Ersetze temporär die Datenabfrage-Methode
    original_method = pipeline._get_real_data
    pipeline._get_real_data = simulate_error
    
    try:
        # Sollte auf Dummy-Daten zurückfallen
        data = pipeline.get_data(1000)
        assert len(data) == 1000
        assert pipeline.validator.validate_structure(data)
    finally:
        # Stelle ursprüngliche Methode wieder her
        pipeline._get_real_data = original_method

@pytest.mark.stress
def test_concurrent_cache_access(pipeline):
    """Test gleichzeitiger Cache-Zugriff"""
    def cache_access():
        try:
            data = pipeline.get_data(1000)
            assert len(data) == 1000
            return True
        except Exception:
            return False
    
    # Starte mehrere Threads für gleichzeitigen Zugriff
    threads = []
    results = []
    
    for _ in range(10):
        thread = threading.Thread(target=lambda: results.append(cache_access()))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Prüfe Ergebnisse
    assert all(results)

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach Tests"""
    yield
    # Lösche Test-Konfiguration und Cache
    import shutil
    from pathlib import Path
    
    paths = ['tests/test_config.yaml', 'tests/cache']
    for path in paths:
        try:
            if Path(path).is_file():
                Path(path).unlink()
            elif Path(path).is_dir():
                shutil.rmtree(path)
        except:
            pass

class TestLoadStressTester:
    def setup_method(self):
        self.tester = LoadStressTester(
            max_concurrent_users=1000,
            duration=timedelta(minutes=10),
            ramp_up_time=timedelta(minutes=1)
        )
        
    def test_initialization(self):
        assert self.tester.is_initialized
        assert self.tester.max_concurrent_users == 1000
        assert self.tester.duration == timedelta(minutes=10)
        
    def test_user_simulation(self):
        users = self.tester.simulate_users(100)
        assert len(users) == 100
        assert all('id' in user for user in users)
        assert all('start_time' in user for user in users)
        
    def test_request_generation(self):
        requests = self.tester.generate_requests(50)
        assert len(requests) == 50
        assert all('type' in req for req in requests)
        assert all('timestamp' in req for req in requests)
        
    def test_high_load_scenario(self):
        results = self.tester.test_high_load()
        assert 'success_rate' in results
        assert 'error_rate' in results
        assert 'response_time' in results
        assert 'throughput' in results
        
    def test_resource_monitoring(self):
        resources = self.tester.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'network_usage' in resources
        assert 'disk_usage' in resources
        
    def test_error_handling(self):
        with pytest.raises(RuntimeError):
            self.tester.simulate_system_failure()
            
    def test_performance_metrics(self):
        metrics = self.tester.get_performance_metrics()
        assert 'requests_per_second' in metrics
        assert 'average_response_time' in metrics
        assert 'error_rate' in metrics
        assert 'resource_utilization' in metrics
        
    def test_report_generation(self):
        report = self.tester.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'recommendations' in report
        assert 'graphs' in report
        
    def test_cleanup(self):
        self.tester.cleanup()
        assert not self.tester.is_initialized 