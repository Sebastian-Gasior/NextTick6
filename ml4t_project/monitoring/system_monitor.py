"""
System-Monitor für das ML4T-Projekt.
"""
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
import os
import platform

class SystemMonitor:
    def __init__(self,
                 check_interval: timedelta = timedelta(seconds=1),
                 alert_threshold: Dict[str, float] = None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialisiert den System-Monitor.
        
        Args:
            check_interval: Intervall für Überprüfungen
            alert_threshold: Schwellenwerte für Alerts
            alert_callback: Callback-Funktion für Alerts
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold or {
            'cpu_usage': 80.0,  # %
            'memory_usage': 80.0,  # %
            'gpu_usage': 80.0,  # %
            'disk_usage': 80.0,  # %
            'network_usage': 80.0,  # %
            'temperature': 80.0  # °C
        }
        self.alert_callback = alert_callback
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()

    def start_monitoring(self):
        """Startet das System-Monitoring."""
        self.start_time = datetime.now()
        self.is_running = True
        self._stop_event.clear()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'disk_usage': [],
            'network_usage': [],
            'temperature': [],
            'process_count': [],
            'alerts': []
        }
        self.logger.info("System-Monitoring gestartet")
        
        # Starte Monitoring-Thread
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stoppt das System-Monitoring."""
        self._stop_event.set()
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join()
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("System-Monitoring beendet")

    def _monitor_loop(self):
        """Monitoring-Hauptschleife."""
        while not self._stop_event.is_set():
            try:
                # Sammle Metriken
                metrics = self._collect_metrics()
                
                # Aktualisiere Metriken-Historie
                for key, value in metrics.items():
                    self.metrics[key].append(value)
                    
                # Prüfe Schwellenwerte
                self._check_thresholds(metrics)
                
                # Warte bis zum nächsten Check
                time.sleep(self.check_interval.total_seconds())
                
            except Exception as e:
                self.logger.error(f"Fehler im Monitoring-Loop: {e}")

    def _collect_metrics(self) -> Dict[str, float]:
        """
        Sammelt aktuelle System-Metriken.
        
        Returns:
            Dictionary mit Metriken
        """
        metrics = {}
        
        # CPU-Metriken
        metrics.update(self._collect_cpu_metrics())
        
        # Speicher-Metriken
        metrics.update(self._collect_memory_metrics())
        
        # GPU-Metriken
        metrics.update(self._collect_gpu_metrics())
        
        # Festplatten-Metriken
        metrics.update(self._collect_disk_metrics())
        
        # Netzwerk-Metriken
        metrics.update(self._collect_network_metrics())
        
        # Prozess-Metriken
        metrics.update(self._collect_process_metrics())
        
        return metrics

    def _collect_cpu_metrics(self) -> Dict[str, float]:
        """
        Sammelt CPU-Metriken.
        
        Returns:
            Dictionary mit CPU-Metriken
        """
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'cpu_frequency': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else 0,
            'cpu_count': psutil.cpu_count()
        }
        
        # CPU-Temperatur (wenn verfügbar)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                cpu_temps = [temp.current for sensor in temps.values() for temp in sensor]
                metrics['temperature'] = np.mean(cpu_temps)
        except Exception:
            metrics['temperature'] = 0.0
            
        return metrics

    def _collect_memory_metrics(self) -> Dict[str, float]:
        """
        Sammelt Speicher-Metriken.
        
        Returns:
            Dictionary mit Speicher-Metriken
        """
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'memory_usage': memory.percent,
            'memory_available': memory.available / 1024 / 1024,  # MB
            'memory_used': memory.used / 1024 / 1024,  # MB
            'swap_usage': swap.percent,
            'swap_used': swap.used / 1024 / 1024  # MB
        }

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """
        Sammelt GPU-Metriken.
        
        Returns:
            Dictionary mit GPU-Metriken
        """
        metrics = {'gpu_usage': 0.0, 'gpu_memory': 0.0}
        
        if torch.cuda.is_available():
            try:
                metrics['gpu_usage'] = torch.cuda.utilization()
                metrics['gpu_memory'] = (
                    torch.cuda.memory_allocated() / 
                    torch.cuda.get_device_properties(0).total_memory * 100
                )
            except Exception as e:
                self.logger.error(f"Fehler bei GPU-Metriken: {e}")
                
        return metrics

    def _collect_disk_metrics(self) -> Dict[str, float]:
        """
        Sammelt Festplatten-Metriken.
        
        Returns:
            Dictionary mit Festplatten-Metriken
        """
        metrics = {}
        
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.update({
                'disk_usage': disk.percent,
                'disk_free': disk.free / 1024 / 1024 / 1024,  # GB
                'disk_read': disk_io.read_bytes / 1024 / 1024,  # MB
                'disk_write': disk_io.write_bytes / 1024 / 1024  # MB
            })
        except Exception as e:
            self.logger.error(f"Fehler bei Festplatten-Metriken: {e}")
            metrics.update({
                'disk_usage': 0.0,
                'disk_free': 0.0,
                'disk_read': 0.0,
                'disk_write': 0.0
            })
            
        return metrics

    def _collect_network_metrics(self) -> Dict[str, float]:
        """
        Sammelt Netzwerk-Metriken.
        
        Returns:
            Dictionary mit Netzwerk-Metriken
        """
        metrics = {}
        
        try:
            net_io = psutil.net_io_counters()
            metrics.update({
                'network_bytes_sent': net_io.bytes_sent / 1024 / 1024,  # MB
                'network_bytes_recv': net_io.bytes_recv / 1024 / 1024,  # MB
                'network_packets_sent': net_io.packets_sent,
                'network_packets_recv': net_io.packets_recv,
                'network_errin': net_io.errin,
                'network_errout': net_io.errout
            })
            
            # Berechne Netzwerkauslastung (sehr vereinfacht)
            total_io = metrics['network_bytes_sent'] + metrics['network_bytes_recv']
            metrics['network_usage'] = min(total_io / 100, 100)  # Maximal 100%
            
        except Exception as e:
            self.logger.error(f"Fehler bei Netzwerk-Metriken: {e}")
            metrics.update({
                'network_usage': 0.0,
                'network_bytes_sent': 0.0,
                'network_bytes_recv': 0.0
            })
            
        return metrics

    def _collect_process_metrics(self) -> Dict[str, float]:
        """
        Sammelt Prozess-Metriken.
        
        Returns:
            Dictionary mit Prozess-Metriken
        """
        metrics = {}
        
        try:
            # Zähle Prozesse
            processes = psutil.process_iter(['name', 'cpu_percent', 'memory_percent'])
            process_count = len(list(processes))
            
            metrics.update({
                'process_count': process_count
            })
            
        except Exception as e:
            self.logger.error(f"Fehler bei Prozess-Metriken: {e}")
            metrics.update({
                'process_count': 0
            })
            
        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]):
        """
        Prüft Metriken gegen Schwellenwerte.
        
        Args:
            metrics: Aktuelle Metriken
        """
        for metric, value in metrics.items():
            if metric in self.alert_threshold:
                threshold = self.alert_threshold[metric]
                if value > threshold:
                    alert = {
                        'timestamp': datetime.now(),
                        'metric': metric,
                        'value': value,
                        'threshold': threshold
                    }
                    self.metrics['alerts'].append(alert)
                    
                    if self.alert_callback:
                        self.alert_callback(alert)
                    
                    self.logger.warning(
                        f"Schwellenwert überschritten: {metric} = "
                        f"{value:.1f} (Schwelle: {threshold:.1f})"
                    )

    def get_system_report(self) -> Dict[str, Any]:
        """
        Generiert einen System-Bericht.
        
        Returns:
            Dictionary mit System-Analyse
        """
        if not self.metrics:
            return {}

        report = {
            'duration': str(self.end_time - self.start_time),
            'system_info': self._get_system_info(),
            'metrics': {},
            'alerts': len(self.metrics['alerts'])
        }

        # Berechne Metriken-Statistiken
        for metric in self.metrics.keys():
            if metric != 'alerts':
                values = np.array(self.metrics[metric])
                report['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p95': float(np.percentile(values, 95))
                }

        # Analysiere Alerts
        if self.metrics['alerts']:
            report['alert_analysis'] = self._analyze_alerts()

        return report

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Sammelt System-Informationen.
        
        Returns:
            Dictionary mit System-Informationen
        """
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'disk_total': psutil.disk_usage('/').total / 1024 / 1024 / 1024  # GB
        }
        
        # GPU-Informationen
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
            })
            
        return info

    def _analyze_alerts(self) -> Dict[str, Any]:
        """
        Analysiert aufgetretene Alerts.
        
        Returns:
            Dictionary mit Alert-Analyse
        """
        analysis = {
            'count_by_metric': {},
            'first_alert': None,
            'last_alert': None,
            'critical_metrics': []
        }
        
        for alert in self.metrics['alerts']:
            # Zähle Alerts pro Metrik
            metric = alert['metric']
            if metric not in analysis['count_by_metric']:
                analysis['count_by_metric'][metric] = 0
            analysis['count_by_metric'][metric] += 1
            
            # Aktualisiere Zeitstempel
            timestamp = alert['timestamp']
            if not analysis['first_alert'] or timestamp < analysis['first_alert']:
                analysis['first_alert'] = timestamp
            if not analysis['last_alert'] or timestamp > analysis['last_alert']:
                analysis['last_alert'] = timestamp
                
        # Identifiziere kritische Metriken
        for metric, count in analysis['count_by_metric'].items():
            if count > 10:
                analysis['critical_metrics'].append(metric)
                
        return analysis

    def get_recommendations(self) -> List[str]:
        """
        Generiert Optimierungsempfehlungen.
        
        Returns:
            Liste von Empfehlungen
        """
        report = self.get_system_report()
        if not report:
            return []

        recommendations = []
        metrics = report['metrics']

        # CPU-Empfehlungen
        if metrics['cpu_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe CPU-Auslastung (P95: {metrics['cpu_usage']['p95']:.1f}%) - "
                "Prozesse optimieren oder CPU-Ressourcen erhöhen"
            )

        # Speicher-Empfehlungen
        if metrics['memory_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe Speicherauslastung (P95: {metrics['memory_usage']['p95']:.1f}%) - "
                "Memory Leaks prüfen oder RAM erweitern"
            )

        # GPU-Empfehlungen
        if 'gpu_usage' in metrics and metrics['gpu_usage']['mean'] > 80:
            recommendations.append(
                f"Hohe GPU-Auslastung (Durchschnitt: {metrics['gpu_usage']['mean']:.1f}%) - "
                "GPU-Workload optimieren"
            )

        # Festplatten-Empfehlungen
        if metrics['disk_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe Festplattenauslastung (P95: {metrics['disk_usage']['p95']:.1f}%) - "
                "Speicherplatz freigeben oder erweitern"
            )

        # Netzwerk-Empfehlungen
        if metrics['network_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe Netzwerkauslastung (P95: {metrics['network_usage']['p95']:.1f}%) - "
                "Netzwerkverkehr optimieren"
            )

        # Prozess-Empfehlungen
        if metrics['process_count']['max'] > 1000:
            recommendations.append(
                f"Viele Prozesse ({metrics['process_count']['max']}) - "
                "Nicht benötigte Prozesse beenden"
            )

        # Temperatur-Empfehlungen
        if 'temperature' in metrics and metrics['temperature']['max'] > 80:
            recommendations.append(
                f"Hohe Systemtemperatur ({metrics['temperature']['max']:.1f}°C) - "
                "Kühlung überprüfen"
            )

        # Alert-basierte Empfehlungen
        if report.get('alert_analysis'):
            for metric in report['alert_analysis'].get('critical_metrics', []):
                recommendations.append(
                    f"Häufige Alerts für {metric} - "
                    "System-Ressourcen überprüfen und optimieren"
                )

        return recommendations

    def cleanup(self):
        """Räumt Monitoring-Ressourcen auf."""
        self.stop_monitoring()
        self.metrics = {} 