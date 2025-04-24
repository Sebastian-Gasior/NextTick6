"""
Last-Tests für das Datenmigrations-System.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import torch
from ..data.data_migration import DataMigrationConfig, DataValidator, HybridDataPipeline
from ml4t_project.testing.migration_stress import MigrationStressTester

@pytest.fixture
def pipeline():
    """Fixture für HybridDataPipeline"""
    return HybridDataPipeline(
        config_path="tests/test_config.yaml",
        cache_dir="tests/cache"
    )

@pytest.fixture
def large_dataset():
    """Fixture für großen Testdatensatz"""
    size = 100000
    dates = pd.date_range(end=datetime.now(), periods=size)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 10, size),
        'high': np.random.normal(105, 10, size),
        'low': np.random.normal(95, 10, size),
        'close': np.random.normal(100, 10, size),
        'volume': np.random.normal(1000000, 100000, size)
    })

@pytest.mark.stress
def test_large_dataset_migration(pipeline, large_dataset):
    """Test Migration großer Datensätze"""
    start_time = time.time()
    
    # Schrittweise Migration
    for _ in range(5):
        pipeline.increase_real_data_ratio()
        data = pipeline.get_mixed_dataset(len(large_dataset))
        
        # Validiere Ergebnisse
        assert len(data) == len(large_dataset)
        assert pipeline.validator.validate_structure(data)
        assert pipeline.validator.validate_data_quality(data)[0]
    
    execution_time = time.time() - start_time
    assert execution_time < 60.0, f"Migration dauerte zu lange: {execution_time:.2f}s"

@pytest.mark.stress
def test_memory_management(pipeline, large_dataset):
    """Test Speichermanagement während Migration"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Führe Migration durch
    for _ in range(3):
        pipeline.increase_real_data_ratio()
        _ = pipeline.get_mixed_dataset(len(large_dataset))
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # Speicherverbrauch sollte kontrolliert bleiben
        assert memory_increase < 2048, f"Zu hoher Speicherverbrauch: {memory_increase:.2f}MB"
        
        # Cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@pytest.mark.stress
def test_concurrent_migrations(pipeline):
    """Test gleichzeitige Migrationen"""
    def worker(size_queue, results):
        while True:
            try:
                size = size_queue.get_nowait()
            except queue.Empty:
                break
                
            try:
                pipeline.increase_real_data_ratio()
                data = pipeline.get_mixed_dataset(size)
                results[size] = len(data)
            except Exception as e:
                results[size] = str(e)
            finally:
                size_queue.task_done()
    
    # Erstelle Warteschlange mit verschiedenen Größen
    sizes = [1000, 5000, 10000, 50000]
    size_queue = queue.Queue()
    for size in sizes:
        size_queue.put(size)
    
    # Starte Worker-Threads
    results = {}
    threads = []
    for _ in range(4):
        thread = threading.Thread(
            target=worker,
            args=(size_queue, results)
        )
        thread.start()
        threads.append(thread)
    
    # Warte auf Fertigstellung
    for thread in threads:
        thread.join()
    
    # Validiere Ergebnisse
    assert len(results) == len(sizes)
    assert all(isinstance(v, int) for v in results.values())
    assert all(results[size] == size for size in sizes)

@pytest.mark.stress
def test_gpu_memory_management(pipeline, large_dataset):
    """Test GPU-Speichermanagement"""
    if not torch.cuda.is_available():
        pytest.skip("GPU nicht verfügbar")
    
    initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # Führe GPU-intensive Operationen durch
    for _ in range(3):
        pipeline.increase_real_data_ratio()
        _ = pipeline.get_mixed_dataset(len(large_dataset))
        
        current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        memory_increase = current_gpu_memory - initial_gpu_memory
        
        # GPU-Speicher sollte kontrolliert bleiben
        assert memory_increase < 4096, f"Zu hoher GPU-Speicherverbrauch: {memory_increase:.2f}MB"
        
        # Cleanup
        torch.cuda.empty_cache()

@pytest.mark.stress
def test_long_running_migration(pipeline):
    """Test Langzeit-Migration"""
    test_duration = 120  # 2 Minuten
    start_time = time.time()
    
    success_count = 0
    error_count = 0
    
    while time.time() - start_time < test_duration:
        try:
            pipeline.increase_real_data_ratio()
            data = pipeline.get_mixed_dataset(10000)
            assert len(data) == 10000
            success_count += 1
        except Exception:
            error_count += 1
        
        time.sleep(1)  # Kurze Pause
    
    # Validiere Stabilität
    total_operations = success_count + error_count
    success_rate = success_count / total_operations if total_operations > 0 else 0
    
    assert success_rate > 0.95, f"Zu viele Fehler: {success_rate:.2%} Erfolgsrate"
    assert success_count > 0, "Keine erfolgreichen Migrationen"

@pytest.mark.stress
def test_data_consistency_during_migration(pipeline, large_dataset):
    """Test Datenkonsistenz während Migration"""
    # Initiale Daten
    initial_data = pipeline.get_mixed_dataset(len(large_dataset))
    initial_stats = initial_data.describe()
    
    # Schrittweise Migration
    for _ in range(5):
        pipeline.increase_real_data_ratio()
        current_data = pipeline.get_mixed_dataset(len(large_dataset))
        current_stats = current_data.describe()
        
        # Statistiken sollten ähnlich bleiben
        pd.testing.assert_frame_equal(
            initial_stats,
            current_stats,
            rtol=0.1  # 10% Toleranz
        )

@pytest.mark.stress
def test_error_recovery_during_migration(pipeline):
    """Test Fehlerbehandlung während Migration"""
    # Simuliere Fehler durch ungültige Daten
    def inject_error():
        raise RuntimeError("Simulierter Fehler")
    
    original_get_real_data = pipeline._get_real_data
    pipeline._get_real_data = inject_error
    
    try:
        # System sollte auf Dummy-Daten zurückfallen
        data = pipeline.get_mixed_dataset(1000)
        assert len(data) == 1000
        assert pipeline.validator.validate_structure(data)
        assert pipeline.validator.validate_data_quality(data)[0]
    finally:
        # Cleanup
        pipeline._get_real_data = original_get_real_data

@pytest.fixture(autouse=True)
def cleanup(pipeline):
    """Cleanup nach Tests"""
    yield
    # Bereinige Cache
    import shutil
    try:
        shutil.rmtree("tests/cache")
    except:
        pass
    
    # Bereinige GPU-Speicher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Warte auf Garbage Collection
    import gc
    gc.collect()
    time.sleep(0.1)

class TestMigrationStressTester:
    def setup_method(self):
        self.tester = MigrationStressTester(
            data_size=1000000,
            batch_size=1000,
            compression=True,
            duration=timedelta(minutes=10)
        )
        
    def test_initialization(self):
        assert self.tester.is_initialized
        assert self.tester.data_size == 1000000
        assert self.tester.batch_size == 1000
        
    def test_data_generation(self):
        data = self.tester.generate_test_data(1000)
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert 'timestamp' in data.columns
        assert 'value' in data.columns
        
    def test_batch_processing(self):
        data = torch.randn(1000, 10)
        processed_data = self.tester.process_batch(data)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape == (1000, 10)
        
    def test_compression_performance(self):
        data = torch.randn(1000, 1000)
        compression_ratio = self.tester.test_compression(data)
        assert isinstance(compression_ratio, float)
        assert compression_ratio > 0
        
    def test_high_volume_migration(self):
        results = self.tester.test_high_volume()
        assert 'success_rate' in results
        assert 'processing_speed' in results
        assert 'error_rate' in results
        
    def test_resource_monitoring(self):
        resources = self.tester.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'disk_usage' in resources
        assert 'network_usage' in resources
        
    def test_error_handling(self):
        with pytest.raises(RuntimeError):
            self.tester.simulate_migration_failure()
            
    def test_performance_metrics(self):
        metrics = self.tester.get_performance_metrics()
        assert 'throughput' in metrics
        assert 'latency' in metrics
        assert 'compression_ratio' in metrics
        assert 'error_rate' in metrics
        
    def test_report_generation(self):
        report = self.tester.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'recommendations' in report
        assert 'graphs' in report
        
    def test_cleanup(self):
        self.tester.cleanup()
        assert not self.tester.is_initialized 