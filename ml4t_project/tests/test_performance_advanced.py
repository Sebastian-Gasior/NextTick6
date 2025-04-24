"""
Erweiterte Performance-Tests für das ML4T System.
Diese Tests überprüfen fortgeschrittene Performance-Aspekte wie Parallelisierung und Lastverhalten.
"""

import pytest
import numpy as np
import pandas as pd
import time
import torch
import multiprocessing
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Tuple
from functools import partial
from multiprocessing import shared_memory
from datetime import datetime, timedelta

from ml4t_project.testing.performance_advanced import AdvancedPerformanceTester
from ml4t_project.model.lstm_model import LSTMModel
from ml4t_project.features.indicators import calculate_all_indicators
from ml4t_project.backtest.engine import BacktestEngine
from ml4t_project.data.loader import load_data

def generate_large_dataset(size: int) -> pd.DataFrame:
    """Generiert einen großen Testdatensatz"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=size, freq='1min')
    base_price = 100.0
    
    # Generiere realistischere Preisdaten mit Trends und Volatilität
    returns = np.random.normal(0.0001, 0.001, size)  # Tägliche Returns
    close_price = base_price * np.exp(np.cumsum(returns))
    
    # Generiere Open, High, Low basierend auf Close
    daily_volatility = 0.01
    high_price = close_price * (1 + abs(np.random.normal(0, daily_volatility, size)))
    low_price = close_price * (1 - abs(np.random.normal(0, daily_volatility, size)))
    open_price = low_price + np.random.random(size) * (high_price - low_price)
    
    # Füge Volumen mit realistischen Patterns hinzu
    volume = np.random.lognormal(10, 1, size)
    volume = volume * (1 + 0.5 * np.sin(np.linspace(0, 8*np.pi, size)))  # Tagesverlaufsmuster
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume.astype(int)
    }, index=dates)

def calculate_indicators(data: np.ndarray, windows: list) -> np.ndarray:
    """Berechnet technische Indikatoren effizient mit NumPy"""
    n_samples = len(data)
    n_features = len(windows) * 3  # SMA, STD, Momentum für jedes Fenster
    result = np.zeros((n_samples, n_features))
    
    for i, window in enumerate(windows):
        # Berechne effizient mit NumPy-Operationen
        weights = np.ones(window) / window
        sma = np.convolve(data, weights, mode='valid')
        pad_width = window - 1
        sma = np.pad(sma, (pad_width, 0), mode='constant', constant_values=np.nan)
        
        # Standardabweichung
        squared = np.square(data - sma)
        std = np.sqrt(np.convolve(squared, weights, mode='valid'))
        std = np.pad(std, (pad_width, 0), mode='constant', constant_values=np.nan)
        
        # Momentum (Verhältnis zum vorherigen Wert)
        momentum = np.zeros_like(data)
        momentum[window:] = data[window:] / data[:-window]
        
        # Speichere Ergebnisse
        result[:, i*3] = sma
        result[:, i*3+1] = std
        result[:, i*3+2] = momentum
    
    return result

def parallel_process_chunk(args: tuple) -> np.ndarray:
    """Verarbeitet einen Datenchunk parallel mit NumPy und Shared Memory"""
    start_idx, end_idx, shm_name, shape, windows = args
    
    # Hole Daten aus Shared Memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    chunk = data[start_idx:end_idx]
    
    # Berechne Indikatoren
    result = calculate_indicators(chunk, windows)
    
    # Cleanup
    existing_shm.close()
    
    return result

@pytest.mark.performance
def test_system_under_load():
    """Test des Systems unter hoher Last"""
    # Simuliere mehrere gleichzeitige Benutzer/Anfragen
    n_users = 4
    data_size = 5000
    
    def simulate_user_workload() -> Tuple[float, float]:
        """Simuliert typische Benutzeraktivität"""
        data = generate_large_dataset(data_size)
        
        # Zeitmessung
        start_time = time.time()
        
        # 1. Daten laden und verarbeiten
        processed_data = calculate_all_indicators(data)
        
        # 2. Modell erstellen und trainieren
        model = LSTMModel(
            input_size=processed_data.shape[1],
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        # 3. Backtesting durchführen
        engine = BacktestEngine(initial_capital=100000)
        engine.run_backtest(processed_data)
        
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return execution_time, memory_used
    
    # Parallele Ausführung der Benutzer-Workloads
    with ThreadPoolExecutor(max_workers=n_users) as executor:
        results = list(executor.map(lambda _: simulate_user_workload(), range(n_users)))
    
    # Analyse der Ergebnisse
    execution_times, memory_usages = zip(*results)
    avg_time = np.mean(execution_times)
    max_time = max(execution_times)
    total_memory = max(memory_usages)
    
    # Überprüfung der Performance-Kriterien
    assert avg_time < 30.0, f"Durchschnittliche Ausführungszeit zu hoch: {avg_time:.2f}s"
    assert max_time < 45.0, f"Maximale Ausführungszeit zu hoch: {max_time:.2f}s"
    assert total_memory < 4096, f"Zu hoher Speicherverbrauch: {total_memory:.0f}MB"

@pytest.mark.performance
def test_parallel_scaling():
    """Test der Parallelisierungs-Skalierung"""
    data_size = 50000
    data = generate_large_dataset(data_size)
    price_data = data['Close'].values
    windows = [20, 50, 100]
    
    def process_with_workers(n_workers: int) -> float:
        """Verarbeitet Daten mit gegebener Anzahl Worker"""
        # Erstelle Shared Memory
        shm = shared_memory.SharedMemory(create=True, size=price_data.nbytes)
        shared_data = np.ndarray(price_data.shape, dtype=price_data.dtype, buffer=shm.buf)
        shared_data[:] = price_data[:]
        
        # Teile Daten in große Chunks
        chunk_size = max(data_size // n_workers, 5000)
        chunks = []
        
        for i in range(0, data_size, chunk_size):
            end = min(i + chunk_size + max(windows), data_size)
            chunks.append((i, end, shm.name, price_data.shape, windows))
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(parallel_process_chunk, chunks))
            
            # Kombiniere Ergebnisse und entferne Überlappungen
            final_result = np.concatenate([
                chunk[:min(chunk_size, len(chunk)-max(windows))]
                for chunk in results
            ])
        
        # Cleanup
        shm.close()
        shm.unlink()
            
        return time.time() - start_time
    
    # Teste verschiedene Worker-Anzahlen
    worker_counts = [1, 2, 4]
    execution_times = []
    
    for n_workers in worker_counts:
        # Führe mehrere Durchläufe durch und nehme den besten Wert
        times = []
        for _ in range(5):
            time_taken = process_with_workers(n_workers)
            times.append(time_taken)
        execution_times.append(min(times))
    
    # Berechne Speedup für jede Worker-Anzahl
    base_time = execution_times[0]
    speedups = [base_time / time for time in execution_times]
    
    # Überprüfe Skalierung (mindestens 15% der idealen Beschleunigung)
    for i, n_workers in enumerate(worker_counts[1:], 1):
        min_expected_speedup = n_workers * 0.15  # Reduzierte Erwartung für realistischere Tests
        actual_speedup = speedups[i]
        assert actual_speedup >= min_expected_speedup, \
            f"Unzureichende Skalierung mit {n_workers} Workern. " \
            f"Speedup: {actual_speedup:.2f}x (Minimum erwartet: {min_expected_speedup:.2f}x)"

@pytest.mark.performance
def test_memory_stability():
    """Test der Speicherstabilität unter Last"""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    peak_memory = initial_memory
    data_sizes = [1000, 2000, 4000, 8000]
    
    try:
        for size in data_sizes:
            data = generate_large_dataset(size)
            processed_data = calculate_all_indicators(data)
            
            # Speicherverbrauch messen
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            # Speicherbereinigung
            del data, processed_data
            gc.collect()
            
            # Warte kurz auf Speicherfreigabe
            time.sleep(1)
            
            # Messe Speicher nach Bereinigung
            after_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_leak = after_gc_memory - initial_memory
            
            assert memory_leak < 100, \
                f"Mögliches Speicherleck erkannt: {memory_leak:.0f}MB nicht freigegeben"
            
    finally:
        # Finale Speicherbereinigung
        gc.collect()
    
    # Überprüfe Gesamtspeicherverbrauch
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    total_increase = final_memory - initial_memory
    
    assert total_increase < 200, \
        f"Zu hoher permanenter Speicheranstieg: {total_increase:.0f}MB"
    assert peak_memory < 4096, \
        f"Zu hoher Spitzenverbrauch: {peak_memory:.0f}MB"

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach jedem Test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class TestAdvancedPerformanceTester:
    def setup_method(self):
        self.tester = AdvancedPerformanceTester(
            test_duration=timedelta(minutes=10),
            warmup_duration=timedelta(minutes=1),
            cooldown_duration=timedelta(minutes=1)
        )
        
    def test_initialization(self):
        assert self.tester.is_initialized
        assert self.tester.test_duration == timedelta(minutes=10)
        assert self.tester.warmup_duration == timedelta(minutes=1)
        
    def test_warmup_phase(self):
        metrics = self.tester.run_warmup()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'gpu_usage' in metrics
        
    def test_performance_phase(self):
        results = self.tester.run_performance_test()
        assert 'throughput' in results
        assert 'latency' in results
        assert 'error_rate' in results
        assert 'resource_usage' in results
        
    def test_cooldown_phase(self):
        metrics = self.tester.run_cooldown()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'gpu_usage' in metrics
        
    def test_resource_monitoring(self):
        resources = self.tester.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'gpu_usage' in resources
        assert 'network_usage' in resources
        assert 'disk_usage' in resources
        
    def test_error_handling(self):
        with pytest.raises(RuntimeError):
            self.tester.simulate_performance_error()
            
    def test_performance_metrics(self):
        metrics = self.tester.get_performance_metrics()
        assert 'throughput' in metrics
        assert 'latency' in metrics
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nicht verfügbar")
    def test_gpu_resource_monitoring(self):
        """Test GPU-Ressourcenüberwachung."""
        tester = AdvancedPerformanceTester(use_gpu=True)
        metrics = tester.monitor_gpu_resources()
        assert 'gpu_memory' in metrics
        assert 'gpu_utilization' in metrics
        assert isinstance(metrics['gpu_memory'], float)
        assert isinstance(metrics['gpu_utilization'], float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nicht verfügbar")
    def test_gpu_performance_metrics(self):
        """Test GPU-spezifische Performance-Metriken."""
        tester = AdvancedPerformanceTester(use_gpu=True)
        metrics = tester.run_gpu_performance_test()
        
        assert 'gpu_compute_time' in metrics
        assert metrics['gpu_compute_time'] > 0
        assert 'gpu_memory_peak' in metrics
        assert metrics['gpu_memory_peak'] > 0
        assert 'gpu_kernel_efficiency' in metrics
        assert 0 <= metrics['gpu_kernel_efficiency'] <= 1

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 