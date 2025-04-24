"""
Langzeitstabilitäts-Tests für das ML4T-System
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import logging
from time import sleep
import psutil
import torch
from ml4t_project.testing.stability import StabilityTester

from ml4t_project.data.loader import load_data
from ml4t_project.features.indicators import add_all_indicators
from ml4t_project.model.predictor import predict
from ml4t_project.signals.logic import generate_signals

# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='stability_test.log'
)
logger = logging.getLogger(__name__)

class DataSimulator:
    """Simuliert kontinuierliche Marktdaten"""
    def __init__(self, initial_price=100, volatility=0.02):
        self.current_price = initial_price
        self.volatility = volatility
        self.is_running = False
        self.queue = queue.Queue()
        
    def start(self):
        """Startet die Datensimulation"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        
    def stop(self):
        """Stoppt die Datensimulation"""
        self.is_running = False
        self.thread.join()
        
    def _run(self):
        """Hauptschleife der Datensimulation"""
        while self.is_running:
            # Simuliere Preisbewegung
            change = np.random.normal(0, self.volatility)
            self.current_price *= (1 + change)
            
            # Erstelle OHLCV-Daten
            data = {
                'Open': self.current_price * (1 + np.random.normal(0, 0.001)),
                'High': self.current_price * (1 + abs(np.random.normal(0, 0.002))),
                'Low': self.current_price * (1 - abs(np.random.normal(0, 0.002))),
                'Close': self.current_price,
                'Volume': np.random.normal(1000000, 100000)
            }
            
            self.queue.put(data)
            sleep(1)  # 1 Sekunde Verzögerung

@pytest.mark.stability
def test_continuous_data_processing(duration_minutes=5):
    """Test der kontinuierlichen Datenverarbeitung"""
    simulator = DataSimulator()
    data_buffer = []
    error_count = 0
    latency_measurements = []
    
    try:
        simulator.start()
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                # Daten sammeln
                data = simulator.queue.get(timeout=1)
                process_start = datetime.now()
                
                # Daten zum Buffer hinzufügen
                data_buffer.append(data)
                
                # Verarbeite Daten wenn genug vorhanden
                if len(data_buffer) >= 100:
                    df = pd.DataFrame(data_buffer)
                    df_with_indicators = add_all_indicators(df)
                    predictions = predict(df_with_indicators)
                    signals = generate_signals(predictions)
                    
                    # Latenz messen
                    process_end = datetime.now()
                    latency = (process_end - process_start).total_seconds()
                    latency_measurements.append(latency)
                    
                    # Buffer aktualisieren
                    data_buffer = data_buffer[-50:]  # Behalte die letzten 50 Datenpunkte
                
            except Exception as e:
                error_count += 1
                logger.error(f"Fehler in der Verarbeitung: {str(e)}")
                
            # Prüfe Fehlerhäufigkeit
            assert error_count < 5, f"Zu viele Fehler aufgetreten: {error_count}"
            
    finally:
        simulator.stop()
    
    # Analysiere Ergebnisse
    avg_latency = np.mean(latency_measurements)
    max_latency = np.max(latency_measurements)
    
    logger.info(f"Durchschnittliche Latenz: {avg_latency:.3f}s")
    logger.info(f"Maximale Latenz: {max_latency:.3f}s")
    logger.info(f"Fehler: {error_count}")
    
    # Überprüfe Performance-Kriterien
    assert avg_latency < 0.1, f"Durchschnittliche Latenz zu hoch: {avg_latency:.3f}s"
    assert max_latency < 0.5, f"Maximale Latenz zu hoch: {max_latency:.3f}s"

@pytest.mark.stability
def test_memory_leak_detection(iterations=1000):
    """Test auf Memory Leaks"""
    initial_memory = psutil.Process().memory_info().rss
    memory_usage = []
    
    for i in range(iterations):
        # Generiere und verarbeite Daten
        df = pd.DataFrame({
            'Open': np.random.normal(100, 10, 100),
            'High': np.random.normal(102, 10, 100),
            'Low': np.random.normal(98, 10, 100),
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.normal(1000000, 100000, 100)
        })
        
        # Führe Operationen aus
        df_with_indicators = add_all_indicators(df)
        predictions = predict(df_with_indicators)
        signals = generate_signals(predictions)
        
        # Speicherverbrauch messen
        current_memory = psutil.Process().memory_info().rss
        memory_usage.append(current_memory)
        
        if i % 100 == 0:
            logger.info(f"Iteration {i}: Memory usage: {current_memory/1024/1024:.2f}MB")
    
    # Analysiere Speicherverbrauch
    memory_growth = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
    logger.info(f"Memory growth rate: {memory_growth/1024/1024:.2f}MB/iteration")
    
    # Überprüfe auf signifikantes Memory Leak
    assert memory_growth < 1024 * 100  # Weniger als 100KB pro Iteration

@pytest.mark.stability
def test_error_recovery():
    """Test der Fehlerbehandlung und Wiederherstellung"""
    error_injected = False
    recovery_successful = False
    
    class ErrorInjector:
        def __init__(self):
            self.original_add_indicators = add_all_indicators
            
        def inject_error(self):
            """Injiziert einen Fehler in add_all_indicators"""
            def faulty_indicators(*args, **kwargs):
                if not error_injected:
                    raise RuntimeError("Simulierter Fehler")
                return self.original_add_indicators(*args, **kwargs)
            
            return faulty_indicators
    
    # Fehler injizieren
    injector = ErrorInjector()
    add_all_indicators = injector.inject_error()
    
    try:
        # Versuche Verarbeitung mit Fehler
        df = pd.DataFrame({
            'Open': np.random.normal(100, 10, 100),
            'High': np.random.normal(102, 10, 100),
            'Low': np.random.normal(98, 10, 100),
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.normal(1000000, 100000, 100)
        })
        
        try:
            _ = add_all_indicators(df)
        except RuntimeError:
            error_injected = True
            
        # Versuche Wiederherstellung
        if error_injected:
            add_all_indicators = injector.original_add_indicators
            df_recovered = add_all_indicators(df)
            recovery_successful = True
            
    finally:
        # Stelle ursprüngliche Funktion wieder her
        add_all_indicators = injector.original_add_indicators
    
    assert error_injected, "Fehler wurde nicht injiziert"
    assert recovery_successful, "Wiederherstellung fehlgeschlagen"

class TestStabilityTester:
    def setup_method(self):
        self.tester = StabilityTester(
            duration=timedelta(hours=24),
            sampling_interval=timedelta(minutes=1)
        )
        
    def test_initialization(self):
        assert self.tester.is_initialized
        assert self.tester.duration == timedelta(hours=24)
        assert self.tester.sampling_interval == timedelta(minutes=1)
        
    def test_long_term_stability(self):
        results = self.tester.test_long_term_stability()
        assert 'uptime' in results
        assert 'error_rate' in results
        assert 'resource_usage' in results
        
    def test_resource_monitoring(self):
        resources = self.tester.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'gpu_usage' in resources
        assert 'disk_usage' in resources
        
    def test_error_handling(self):
        with pytest.raises(RuntimeError):
            self.tester.simulate_stability_error()
            
    def test_performance_metrics(self):
        metrics = self.tester.get_performance_metrics()
        assert 'uptime' in metrics
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