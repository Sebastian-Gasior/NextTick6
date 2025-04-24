"""
Load Stress Testing für das ML4T-Projekt.
"""
import time
import threading
import numpy as np
from typing import Dict, Optional, Any, List, Callable
import psutil
import torch
import logging
from datetime import datetime, timedelta

class LoadStressTester:
    """Klasse für Load Stress Testing."""
    
    def __init__(self,
                 test_duration: timedelta = timedelta(minutes=5),
                 ramp_up_time: timedelta = timedelta(seconds=30),
                 cool_down_time: timedelta = timedelta(seconds=30),
                 target_load: float = 0.8,
                 check_interval: float = 1.0,
                 metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None):
        """
        Initialisiert den LoadStressTester.
        
        Args:
            test_duration: Gesamtdauer des Tests
            ramp_up_time: Zeit für graduelles Hochfahren
            cool_down_time: Zeit für graduelles Herunterfahren
            target_load: Ziel-CPU-Last (0-1)
            check_interval: Intervall für Metrik-Checks
            metrics_callback: Optionale Callback-Funktion für Metriken
        """
        self.test_duration = test_duration
        self.ramp_up_time = ramp_up_time
        self.cool_down_time = cool_down_time
        self.target_load = target_load
        self.check_interval = check_interval
        self.metrics_callback = metrics_callback
        
        self.stop_event = threading.Event()
        self.metrics: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'response_time': [],
            'error_rate': []
        }
        self.logger = logging.getLogger(__name__)
        
    def start_test(self) -> None:
        """Startet den Load Stress Test."""
        self.logger.info("Starte Load Stress Test")
        self.stop_event.clear()
        
        # Starte Monitoring Thread
        monitor_thread = threading.Thread(target=self._monitor_metrics)
        monitor_thread.start()
        
        try:
            # Ramp-up Phase
            self._execute_ramp_up()
            
            # Haupttest-Phase
            self._execute_main_test()
            
            # Cool-down Phase
            self._execute_cool_down()
            
        except Exception as e:
            self.logger.error(f"Fehler während Load Stress Test: {str(e)}")
            raise
        
        finally:
            self.stop_event.set()
            monitor_thread.join()
            
    def stop_test(self) -> None:
        """Stoppt den Load Stress Test."""
        self.logger.info("Stoppe Load Stress Test")
        self.stop_event.set()
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Gibt die gesammelten Metriken zurück.
        
        Returns:
            Dictionary mit Metrik-Listen
        """
        return self.metrics
        
    def _monitor_metrics(self) -> None:
        """Überwacht System-Metriken."""
        while not self.stop_event.is_set():
            cpu_usage = psutil.cpu_percent() / 100.0
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # GPU-Metriken falls verfügbar
            gpu_usage = 0.0
            if torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                except Exception as e:
                    self.logger.warning(f"Fehler beim Sammeln von GPU-Metriken: {str(e)}")
            
            # Sammle Metriken
            self.metrics['cpu_usage'].append(cpu_usage)
            self.metrics['memory_usage'].append(memory_usage)
            self.metrics['gpu_usage'].append(gpu_usage)
            
            # Callback falls vorhanden
            if self.metrics_callback:
                current_metrics = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'gpu_usage': gpu_usage
                }
                self.metrics_callback(current_metrics)
            
            time.sleep(self.check_interval)
            
    def _execute_ramp_up(self) -> None:
        """Führt die Ramp-up Phase aus."""
        self.logger.info("Starte Ramp-up Phase")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < self.ramp_up_time and not self.stop_event.is_set():
            progress = (datetime.now() - start_time) / self.ramp_up_time
            current_target = self.target_load * progress
            self._generate_load(current_target)
            time.sleep(0.1)
            
    def _execute_main_test(self) -> None:
        """Führt die Haupttest-Phase aus."""
        self.logger.info("Starte Haupttest-Phase")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < self.test_duration and not self.stop_event.is_set():
            self._generate_load(self.target_load)
            time.sleep(0.1)
            
    def _execute_cool_down(self) -> None:
        """Führt die Cool-down Phase aus."""
        self.logger.info("Starte Cool-down Phase")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < self.cool_down_time and not self.stop_event.is_set():
            progress = 1 - (datetime.now() - start_time) / self.cool_down_time
            current_target = self.target_load * progress
            self._generate_load(current_target)
            time.sleep(0.1)
            
    def _generate_load(self, target_load: float) -> None:
        """
        Generiert CPU-Last.
        
        Args:
            target_load: Ziel-CPU-Last (0-1)
        """
        start_time = time.time()
        while (time.time() - start_time) < self.check_interval:
            if target_load > 0:
                # CPU-intensive Operation
                _ = np.random.random((1000, 1000)) @ np.random.random((1000, 1000))
            time.sleep(max(0, (1 - target_load) * self.check_interval))
            
    def get_test_report(self) -> Dict[str, Any]:
        """
        Generiert einen Testbericht.
        
        Returns:
            Dictionary mit Testbericht
        """
        report = {
            'duration': self.test_duration.total_seconds(),
            'target_load': self.target_load,
            'metrics': {
                'cpu_usage': {
                    'mean': np.mean(self.metrics['cpu_usage']),
                    'max': np.max(self.metrics['cpu_usage']),
                    'min': np.min(self.metrics['cpu_usage']),
                    'std': np.std(self.metrics['cpu_usage'])
                },
                'memory_usage': {
                    'mean': np.mean(self.metrics['memory_usage']),
                    'max': np.max(self.metrics['memory_usage']),
                    'min': np.min(self.metrics['memory_usage']),
                    'std': np.std(self.metrics['memory_usage'])
                },
                'gpu_usage': {
                    'mean': np.mean(self.metrics['gpu_usage']),
                    'max': np.max(self.metrics['gpu_usage']),
                    'min': np.min(self.metrics['gpu_usage']),
                    'std': np.std(self.metrics['gpu_usage'])
                }
            }
        }
        return report 