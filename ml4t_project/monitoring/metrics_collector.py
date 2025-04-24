"""
Metrics Collector für System- und Performance-Monitoring.
"""
import os
import time
import psutil
import yaml
import json
from datetime import datetime
import logging
from pathlib import Path

class MetricsCollector:
    """Sammelt System- und Performance-Metriken."""
    
    def __init__(self, config_path="config/monitoring.yaml"):
        """Initialisiert den MetricsCollector."""
        self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
    def load_config(self, config_path):
        """Lädt die Monitoring-Konfiguration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['monitoring']
            
    def setup_logging(self):
        """Richtet das Logging-System ein."""
        log_dir = Path(self.config['log_dir']) / 'monitoring'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            filename=str(log_file),
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Erstellt die notwendigen Verzeichnisse."""
        metrics_dir = Path(self.config['metrics_dir']) / 'current'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = metrics_dir
        
    def collect_metrics(self):
        """Sammelt aktuelle System-Metriken."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'error_rate': 0.0  # Wird später implementiert
        }
        
        # GPU-Metriken hinzufügen, wenn verfügbar
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_usage'] = gpus[0].load * 100
        except ImportError:
            metrics['gpu_usage'] = 0.0
            
        return metrics
        
    def check_thresholds(self, metrics):
        """Überprüft die Metriken gegen definierte Schwellenwerte."""
        for metric, value in metrics.items():
            if metric in self.config['thresholds']:
                threshold = self.config['thresholds'][metric]
                if value > threshold:
                    logging.warning(
                        f"Schwellenwert überschritten: {metric} = {value:.1f}% "
                        f"(Schwellenwert: {threshold}%)"
                    )
                    
    def save_metrics(self, metrics):
        """Speichert die Metriken."""
        current_file = self.metrics_dir / 'current_metrics.json'
        with open(current_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def start_collection(self, interval=60):
        """Startet die Metrik-Sammlung."""
        logging.info("Starte Metrik-Sammlung...")
        print("Metrik-Sammlung gestartet. Drücken Sie Ctrl+C zum Beenden.")
        
        try:
            while True:
                metrics = self.collect_metrics()
                self.check_thresholds(metrics)
                self.save_metrics(metrics)
                
                print(f"\rAktuelle Metriken - CPU: {metrics['cpu_usage']:.1f}%, "
                      f"RAM: {metrics['memory_usage']:.1f}%, "
                      f"Disk: {metrics['disk_usage']:.1f}%", end='')
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logging.info("Metrik-Sammlung beendet durch Benutzer.")
            print("\nMetrik-Sammlung beendet.") 