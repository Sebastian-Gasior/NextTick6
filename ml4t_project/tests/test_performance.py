"""
Performance und Last-Tests für das ML4T-System
"""
import pytest
import numpy as np
import pandas as pd
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import tensorflow as tf
import torch
from typing import List, Dict, Tuple, Any
import gc
from datetime import datetime, timedelta

from ml4t_project.testing.performance import PerformanceTester
from ml4t_project.features.indicators import add_all_indicators
from ml4t_project.model.builder import build_lstm_model
from ml4t_project.model.trainer import train_model
from ml4t_project.data.loader import load_data

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach jedem Test"""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if tf.test.is_built_with_cuda():
        tf.keras.backend.clear_session()

def generate_test_data(size: int) -> pd.DataFrame:
    """Generiert synthetische Testdaten"""
    np.random.seed(42)  # Für Reproduzierbarkeit
    dates = pd.date_range(end=pd.Timestamp.now(), periods=size)
    data = {
        'Open': np.random.normal(100, 10, size),
        'High': np.random.normal(102, 10, size),
        'Low': np.random.normal(98, 10, size),
        'Close': np.random.normal(100, 10, size),
        'Volume': np.random.randint(1000000, 10000000, size)
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure High > Low
    df['High'] = df[['High', 'Low']].max(axis=1)
    df['Low'] = df[['High', 'Low']].min(axis=1)
    return df

def measure_performance(func, *args, **kwargs) -> Tuple[float, float, Any]:
    """
    Misst Ausführungszeit und Speicherverbrauch einer Funktion.
    
    Returns:
        Tuple aus (Zeit, Speicheränderung, Funktionsergebnis)
    """
    gc.collect()  # Speicherbereinigung vor der Messung
    
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time()
    
    result = func(*args, **kwargs)
    
    end_time = time()
    end_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    return end_time - start_time, end_mem - start_mem, result

def process_symbol_data(symbol: str) -> pd.DataFrame:
    """Verarbeitet Daten für ein einzelnes Symbol"""
    df = generate_test_data(5000)  # Größere Datenmenge
    # Simuliere I/O-Verzögerung
    sleep(0.1)  # Realistische I/O-Verzögerung
    return add_all_indicators(df)

@pytest.mark.performance
def test_indicator_calculation_performance():
    """Test der Performance der Indikator-Berechnung"""
    sizes = [1000, 2000, 4000]  # Kleinere Schritte für bessere Skalierung
    metrics: List[Dict] = []

    for size in sizes:
        df = generate_test_data(size)
        exec_time, mem_usage, results = measure_performance(add_all_indicators, df)
        
        # Validierung der Ergebnisse
        assert isinstance(results, pd.DataFrame), "Ergebnis muss DataFrame sein"
        assert not results.empty, "Ergebnis darf nicht leer sein"
        assert not results.isnull().all().any(), "Keine komplett leeren Spalten erlaubt"
        
        metrics.append({
            'size': size,
            'time': exec_time,
            'memory': mem_usage,
            'time_per_row': exec_time / size,
            'memory_per_row': mem_usage / size
        })
    
    # Überprüfe Skalierung mit angepasster Toleranz
    for i in range(len(metrics)-1):
        time_ratio = metrics[i+1]['time'] / metrics[i]['time']
        size_ratio = metrics[i+1]['size'] / metrics[i]['size']
        scaling_factor = time_ratio / size_ratio
        assert 0.5 <= scaling_factor <= 2.0, f"Nicht-lineare Skalierung: {scaling_factor:.2f}"

@pytest.mark.performance
def test_parallel_data_processing():
    """Test der parallelen Datenverarbeitung"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Reduzierte Anzahl für klarere Ergebnisse
    
    # Sequentielle Verarbeitung
    seq_start = time()
    sequential_results = [process_symbol_data(symbol) for symbol in symbols]
    seq_time = time() - seq_start
    
    # Thread-basierte Parallelisierung mit Chunk-Größe
    with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as executor:
        thread_start = time()
        thread_results = list(executor.map(process_symbol_data, symbols))
        thread_time = time() - thread_start
    
    # Prozess-basierte Parallelisierung mit Chunk-Größe
    with ProcessPoolExecutor(max_workers=min(4, len(symbols))) as executor:
        process_start = time()
        process_results = list(executor.map(process_symbol_data, symbols))
        process_time = time() - process_start
    
    # Validierung der Ergebnisse
    for results in [sequential_results, thread_results, process_results]:
        assert len(results) == len(symbols), "Falsche Anzahl an Ergebnissen"
        for df in results:
            assert isinstance(df, pd.DataFrame), "Ungültiges Ergebnis-Format"
            assert not df.empty, "Leeres DataFrame"
    
    # Performance-Vergleich mit angepassten Erwartungen
    best_parallel_time = min(thread_time, process_time)
    speedup = seq_time / best_parallel_time
    assert speedup > 1.1, f"Unzureichende Parallelisierung: {speedup:.2f}x Speedup"

@pytest.mark.performance
def test_model_memory_usage():
    """Test des Speicherverbrauchs des LSTM-Modells"""
    configs = [
        {'sequence_length': 20, 'n_features': 10, 'n_units': 50},
        {'sequence_length': 30, 'n_features': 15, 'n_units': 100}
    ]
    
    for config in configs:
        # Modell erstellen und Speicherverbrauch messen
        _, mem_usage, model = measure_performance(
            build_lstm_model,
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            n_units=config['n_units']
        )
        
        # Validierung des Modells
        assert isinstance(model, tf.keras.Model), "Ungültiges Modell"
        
        # Überprüfe Modellarchitektur
        expected_input_shape = (None, config['sequence_length'], config['n_features'])
        assert model.input_shape == expected_input_shape, "Falsche Eingabe-Form"
        
        # Speicherlimits basierend auf Modellgröße
        max_memory = 100 * (config['n_units'] / 50)  # Skaliert mit Modellgröße
        assert mem_usage < max_memory, f"Zu hoher Speicherverbrauch: {mem_usage:.2f}MB"

@pytest.mark.performance
def test_gpu_acceleration():
    """Test der GPU-Beschleunigung (falls verfügbar)"""
    if not tf.test.is_built_with_cuda():
        pytest.skip("Keine GPU verfügbar")
    
    # Testdaten
    batch_size = 32
    X_train = np.random.normal(0, 1, (1000, 20, 10))
    y_train = np.random.normal(0, 1, (1000, 1))
    X_val = X_train[:200]
    y_val = y_train[:200]
    
    # CPU Training
    with tf.device('/CPU:0'):
        cpu_time, cpu_mem, history_cpu = measure_performance(
            lambda: train_model(
                build_lstm_model(20, 10),
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=5,
                batch_size=batch_size
            )
        )
    
    # GPU Training
    with tf.device('/GPU:0'):
        gpu_time, gpu_mem, history_gpu = measure_performance(
            lambda: train_model(
                build_lstm_model(20, 10),
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=5,
                batch_size=batch_size
            )
        )
    
    # Validierung der Trainingsergebnisse
    for history in [history_cpu, history_gpu]:
        assert isinstance(history, dict), "Ergebnis muss ein Dictionary sein"
        assert 'loss' in history, "Kein Loss aufgezeichnet"
        assert all(loss > 0 for loss in history['loss']), "Ungültige Loss-Werte"
    
    # Performance-Vergleich
    speedup = cpu_time / gpu_time
    assert speedup > 2.0, f"Unzureichende GPU-Beschleunigung: {speedup:.2f}x"

@pytest.mark.performance
def test_batch_processing_optimization():
    """Test der Batch-Verarbeitung Optimierung"""
    X_train = np.random.normal(0, 1, (1000, 20, 10))
    y_train = np.random.normal(0, 1, (1000, 1))
    X_val = X_train[:200]
    y_val = y_train[:200]
    batch_sizes = [16, 32, 64, 128]
    metrics: List[Dict] = []
    
    model = build_lstm_model(20, 10)
    
    for batch_size in batch_sizes:
        exec_time, mem_usage, history = measure_performance(
            train_model,
            model, X_train, y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=5,
            batch_size=batch_size
        )
        
        # Validierung des Trainings
        assert isinstance(history, dict), "Ergebnis muss ein Dictionary sein"
        assert 'loss' in history, "Kein Loss aufgezeichnet"
        losses = history['loss']
        assert all(loss > 0 for loss in losses), "Ungültige Loss-Werte"
        
        metrics.append({
            'batch_size': batch_size,
            'time': exec_time,
            'memory': mem_usage,
            'final_loss': losses[-1],
            'loss_improvement': (losses[0] - losses[-1]) / losses[0]
        })
    
    # Finde optimale Batch-Größe
    optimal_idx = np.argmin([m['time'] for m in metrics])
    optimal_batch_size = batch_sizes[optimal_idx]
    
    # Validierung der Optimierung
    assert optimal_batch_size > batch_sizes[0], "Zu kleine optimale Batch-Größe"
    
    # Überprüfe Speichereffizienz
    for i in range(len(metrics)-1):
        mem_ratio = metrics[i+1]['memory'] / metrics[i]['memory']
        assert mem_ratio < 2.0, f"Zu hoher Speicherzuwachs bei Batch-Size {batch_sizes[i+1]}"
        
        # Überprüfe Loss-Stabilität
        loss_ratio = metrics[i+1]['final_loss'] / metrics[i]['final_loss']
        assert 0.5 < loss_ratio < 2.0, "Zu große Loss-Variation zwischen Batch-Sizes"

class TestPerformanceTester:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Test Setup mit grundlegender Konfiguration"""
        self.config = {
            'warmup_duration': timedelta(seconds=2),
            'test_duration': timedelta(seconds=5),
            'cooldown_duration': timedelta(seconds=2),
            'metrics_interval': timedelta(seconds=1),
            'resource_check_interval': timedelta(seconds=1),
            'error_threshold': 0.05,
            'performance_threshold': {
                'latency_ms': 100,
                'throughput_ops': 1000
            }
        }
        self.tester = PerformanceTester(self.config)
        yield
        self.tester.cleanup()

    def test_initialization(self):
        """Testet die korrekte Initialisierung des Performance Testers"""
        assert self.tester is not None
        assert isinstance(self.tester.config, dict)
        assert self.tester.metrics == {}
        assert not self.tester.is_running

    def test_start_stop(self):
        """Testet Start- und Stop-Funktionalität"""
        assert not self.tester.is_running
        self.tester.start()
        assert self.tester.is_running
        assert self.tester.start_time is not None
        sleep(0.1)
        self.tester.stop()
        assert not self.tester.is_running
        assert self.tester.end_time is not None
        assert self.tester.end_time > self.tester.start_time

    def test_metric_collection(self):
        """Testet die Sammlung von Performance-Metriken"""
        self.tester.start()
        sleep(2)  # Sammle einige Metriken
        metrics = self.tester.get_metrics()
        self.tester.stop()

        assert isinstance(metrics, dict)
        required_metrics = ['cpu_usage', 'memory_usage', 'latency', 'throughput']
        for metric in required_metrics:
            assert metric in metrics, f"Metrik {metric} fehlt in den Ergebnissen"
            assert isinstance(metrics[metric], (int, float, list, np.ndarray))

    def test_resource_monitoring(self):
        """Testet das Monitoring von Systemressourcen"""
        self.tester.start()
        sleep(1)
        resources = self.tester.get_resource_usage()
        self.tester.stop()

        assert isinstance(resources, dict)
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert isinstance(resources['cpu_usage'], float)
        assert 0 <= resources['cpu_usage'] <= 100
        assert isinstance(resources['memory_usage'], float)
        assert resources['memory_usage'] > 0

    def test_error_handling(self):
        """Testet die Fehlerbehandlung"""
        with pytest.raises(ValueError):
            PerformanceTester({'invalid_config': True})

        with pytest.raises(RuntimeError):
            self.tester.stop()  # Versuche zu stoppen ohne zu starten

    @pytest.mark.parametrize("load_level", ["low", "medium", "high"])
    def test_different_load_levels(self, load_level):
        """Testet verschiedene Lastszenarien"""
        config = self.config.copy()
        config['load_level'] = load_level
        tester = PerformanceTester(config)
        
        tester.start()
        sleep(2)
        metrics = tester.get_metrics()
        tester.stop()

        assert 'throughput' in metrics
        assert 'latency' in metrics
        if load_level == "high":
            assert metrics['cpu_usage'] > 50  # Erwarte höhere CPU-Auslastung bei hoher Last

    def test_performance_thresholds(self):
        """Testet die Überprüfung von Performance-Schwellenwerten"""
        self.tester.start()
        sleep(2)
        metrics = self.tester.get_metrics()
        self.tester.stop()

        assert metrics['latency'] <= self.config['performance_threshold']['latency_ms'], \
            "Latenz überschreitet den Schwellenwert"
        assert metrics['throughput'] >= self.config['performance_threshold']['throughput_ops'], \
            "Durchsatz unterschreitet den Schwellenwert"

    def test_report_generation(self):
        """Testet die Generierung von Performance-Berichten"""
        self.tester.start()
        sleep(2)
        self.tester.stop()
        
        report = self.tester.generate_report()
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'detailed_metrics' in report
        assert 'test_duration' in report
        assert isinstance(report['summary'], dict)
        assert isinstance(report['detailed_metrics'], dict)
        assert isinstance(report['test_duration'], float)

    def test_cleanup(self):
        """Testet die Cleanup-Funktionalität"""
        self.tester.start()
        sleep(1)
        self.tester.stop()
        self.tester.cleanup()
        
        assert self.tester.metrics == {}
        assert not self.tester.is_running
        assert self.tester.start_time is None
        assert self.tester.end_time is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nicht verfügbar")
    def test_gpu_performance(self):
        """Test GPU-spezifische Performance-Metriken."""
        tester = PerformanceTester(use_gpu=True)
        metrics = tester.measure_gpu_performance()
        assert 'gpu_utilization' in metrics
        assert 'gpu_memory_used' in metrics
        assert metrics['gpu_utilization'] >= 0
        assert metrics['gpu_memory_used'] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU nicht verfügbar")
    def test_gpu_resource_monitoring(self):
        """Test GPU-Ressourcenüberwachung."""
        tester = PerformanceTester(use_gpu=True)
        metrics = tester.monitor_resources()
        
        assert 'gpu_memory' in metrics
        assert isinstance(metrics['gpu_memory'], float)
        assert 'gpu_utilization' in metrics
        assert isinstance(metrics['gpu_utilization'], float)

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 