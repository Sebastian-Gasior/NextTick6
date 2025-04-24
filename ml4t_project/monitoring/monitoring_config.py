"""
Monitoring-Konfiguration für das ML4T-Projekt.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from datetime import timedelta

@dataclass
class MonitoringThresholds:
    """Schwellenwerte für Monitoring-Alerts."""
    cpu_usage: float = 80.0  # Prozent
    memory_usage: float = 80.0  # Prozent
    gpu_usage: float = 80.0  # Prozent
    disk_usage: float = 80.0  # Prozent
    latency: float = 1.0  # Sekunden
    error_rate: float = 0.01  # 1%

@dataclass
class MonitoringConfig:
    """Monitoring-Konfiguration."""
    enabled: bool = True
    log_level: str = "INFO"
    check_interval: timedelta = timedelta(seconds=60)
    metrics_retention: timedelta = timedelta(days=30)
    log_dir: str = "logs"
    metrics_dir: str = "metrics"
    thresholds: MonitoringThresholds = MonitoringThresholds()

class MonitoringConfigLoader:
    """Lädt und verwaltet die Monitoring-Konfiguration."""
    
    def __init__(self, config_path: str = "config/monitoring.yaml"):
        """
        Initialisiert den Config-Loader.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        
    def _load_config(self) -> MonitoringConfig:
        """Lädt die Konfiguration."""
        if not self.config_path.exists():
            self.logger.warning(f"Keine Konfiguration gefunden unter {self.config_path}")
            self.logger.info("Erstelle Standard-Konfiguration")
            self._create_default_config()
            
        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)
            
        return self._parse_config(config_dict)
    
    def _create_default_config(self):
        """Erstellt Standard-Konfiguration."""
        config = {
            'monitoring': {
                'enabled': True,
                'log_level': 'INFO',
                'check_interval_seconds': 60,
                'metrics_retention_days': 30,
                'log_dir': 'logs',
                'metrics_dir': 'metrics',
                'thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 80.0,
                    'gpu_usage': 80.0,
                    'disk_usage': 80.0,
                    'latency': 1.0,
                    'error_rate': 0.01
                }
            }
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
    def _parse_config(self, config_dict: Dict[str, Any]) -> MonitoringConfig:
        """
        Parsed die Konfiguration.
        
        Args:
            config_dict: Dictionary mit Konfiguration
            
        Returns:
            MonitoringConfig-Instanz
        """
        monitoring = config_dict.get('monitoring', {})
        
        thresholds = MonitoringThresholds(
            **monitoring.get('thresholds', {})
        )
        
        return MonitoringConfig(
            enabled=monitoring.get('enabled', True),
            log_level=monitoring.get('log_level', 'INFO'),
            check_interval=timedelta(
                seconds=monitoring.get('check_interval_seconds', 60)
            ),
            metrics_retention=timedelta(
                days=monitoring.get('metrics_retention_days', 30)
            ),
            log_dir=monitoring.get('log_dir', 'logs'),
            metrics_dir=monitoring.get('metrics_dir', 'metrics'),
            thresholds=thresholds
        )
        
    def get_config(self) -> MonitoringConfig:
        """
        Gibt die aktuelle Konfiguration zurück.
        
        Returns:
            MonitoringConfig-Instanz
        """
        return self.config
        
    def update_config(self, new_config: Dict[str, Any]):
        """
        Aktualisiert die Konfiguration.
        
        Args:
            new_config: Neue Konfigurationswerte
        """
        config_dict = {
            'monitoring': new_config
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
            
        self.config = self._parse_config(config_dict)
        self.logger.info("Konfiguration aktualisiert") 