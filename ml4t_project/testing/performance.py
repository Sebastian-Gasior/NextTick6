import time
import logging
import psutil
from datetime import timedelta
import torch

class PerformanceTester:
    """Klasse für grundlegende Performance-Tests und Ressourcenüberwachung."""
    
    def __init__(self, 
                 test_duration: timedelta = timedelta(minutes=5),
                 warmup_duration: timedelta = timedelta(seconds=30),
                 cooldown_duration: timedelta = timedelta(seconds=30)):
        """
        Initialisiert den Performance Tester.
        
        Args:
            test_duration: Dauer des Haupttests
            warmup_duration: Dauer der Aufwärmphase
            cooldown_duration: Dauer der Abkühlphase
        """
        self.test_duration = test_duration
        self.warmup_duration = warmup_duration 
        self.cooldown_duration = cooldown_duration
        
        # Logger Setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Performance Metriken
        self.metrics = {
            'throughput': 0.0,
            'latency': 0.0,
            'error_rate': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0 if torch.cuda.is_available() else None
        }

    def initialize(self):
        """Initialisiert die Testumgebung."""
        self.logger.info("Initialisiere Performance Test...")
        self.metrics = {
            'throughput': 0.0,  # Operationen pro Sekunde
            'latency': 0.0,     # Millisekunden
            'error_rate': 0.0,  # Prozent
            'cpu_usage': 0.0,   # Prozent
            'memory_usage': 0.0, # Prozent
            'gpu_usage': 0.0 if torch.cuda.is_available() else None  # Prozent
        }
        return True

    def measure_performance(self, num_operations: int, duration: float) -> dict:
        """
        Misst die Performance-Metriken.
        
        Args:
            num_operations: Anzahl der durchgeführten Operationen
            duration: Zeitdauer in Sekunden
            
        Returns:
            Dict mit Performance-Metriken
        """
        # Berechne Durchsatz
        throughput = num_operations / duration if duration > 0 else 0
        
        # Erfasse Ressourcennutzung
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # GPU Nutzung falls verfügbar
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            
        metrics = {
            'throughput': throughput,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage
        }
        
        self.logger.info(f"Performance Metriken: {metrics}")
        return metrics

    def monitor_resources(self) -> dict:
        """
        Überwacht die Systemressourcen.
        
        Returns:
            Dict mit Ressourcennutzung
        """
        resources = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            resources['gpu_usage'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            
        return resources

    def cleanup(self):
        """Räumt Ressourcen nach dem Test auf."""
        self.logger.info("Räume Ressourcen auf...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True 