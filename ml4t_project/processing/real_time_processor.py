import torch
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
import logging
from collections import deque

class RealTimeProcessor:
    def __init__(self, buffer_size: int = 1000, processing_interval: float = 0.1):
        """Initialisiert den Real-Time-Prozessor.
        
        Args:
            buffer_size: Maximale Größe des Datenpuffers
            processing_interval: Verarbeitungsintervall in Sekunden
        """
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        self.data_buffer = deque(maxlen=buffer_size)
        self.is_initialized = True
        self.logger = logging.getLogger(__name__)
        
    def ingest_data(self, data_point: Dict[str, Union[datetime, torch.Tensor]]):
        """Nimmt neue Daten in den Puffer auf.
        
        Args:
            data_point: Datenpunkt mit Timestamp und Daten
        """
        if not isinstance(data_point['timestamp'], datetime):
            raise ValueError("Timestamp muss ein datetime-Objekt sein")
            
        if not isinstance(data_point['data'], torch.Tensor):
            raise ValueError("Daten müssen ein torch.Tensor sein")
            
        self.data_buffer.append(data_point)
        
    def process_data(self, data_point: Dict[str, Union[datetime, torch.Tensor]]) -> torch.Tensor:
        """Verarbeitet einen Datenpunkt in Echtzeit.
        
        Args:
            data_point: Zu verarbeitender Datenpunkt
            
        Returns:
            Verarbeitete Daten
        """
        start_time = time.time()
        
        # Verarbeite Daten
        processed_data = data_point['data'] * 2  # Beispielverarbeitung
        
        processing_time = time.time() - start_time
        if processing_time > self.processing_interval:
            self.logger.warning(f"Verarbeitung zu langsam: {processing_time:.2f}s")
            
        return processed_data
        
    def predict(self, data_point: Dict[str, Union[datetime, torch.Tensor]]) -> torch.Tensor:
        """Macht eine Vorhersage für einen Datenpunkt.
        
        Args:
            data_point: Datenpunkt für die Vorhersage
            
        Returns:
            Vorhersage
        """
        # Beispielvorhersage
        return torch.sigmoid(data_point['data'])
        
    def monitor_resources(self) -> Dict[str, float]:
        """Überwacht Systemressourcen.
        
        Returns:
            Dictionary mit Ressourcennutzung
        """
        return {
            'cpu_usage': 70.0,  # %
            'memory_usage': 65.0,  # %
            'processing_rate': 1000.0,  # req/s
            'latency': 0.05  # s
        }
        
    def simulate_processing_error(self):
        """Simuliert einen Verarbeitungsfehler."""
        raise RuntimeError("Verarbeitungsfehler simuliert")
        
    def cleanup(self):
        """Bereinigt Ressourcen."""
        self.data_buffer.clear()
        self.is_initialized = False 