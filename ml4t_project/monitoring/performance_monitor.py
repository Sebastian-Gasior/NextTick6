"""
Performance-Monitor für das ML4T-Projekt.
"""
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch

class PerformanceMonitor:
    def __init__(self,
                 check_interval: timedelta = timedelta(seconds=1),
                 alert_threshold: Dict[str, float] = None,
                 alert_callback: Optional[Callable] = None):
        """
        Initialisiert den Performance-Monitor.
        
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
            'latency': 1.0  # Sekunden
        }
        self.alert_callback = alert_callback
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()

    def start_monitoring(self):
        """Startet das Performance-Monitoring."""
        self.start_time = datetime.now()
        self.is_running = True
        self._stop_event.clear()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'disk_usage': [],
            'latency': [],
            'alerts': []
        }
        self.logger.info("Performance-Monitoring gestartet")
        
        # Starte Monitoring-Thread
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stoppt das Performance-Monitoring."""
        self._stop_event.set()
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join()
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("Performance-Monitoring beendet")

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
        Sammelt aktuelle Performance-Metriken.
        
        Returns:
            Dictionary mit Metriken
        """
        metrics = {}
        
        # CPU-Auslastung
        metrics['cpu_usage'] = psutil.cpu_percent()
        
        # Speicherauslastung
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        
        # GPU-Auslastung
        if torch.cuda.is_available():
            metrics['gpu_usage'] = torch.cuda.utilization()
        else:
            metrics['gpu_usage'] = 0.0
            
        # Festplattenauslastung
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = disk.percent
        
        # Latenz (simuliert)
        metrics['latency'] = self._measure_latency()
        
        return metrics

    def _measure_latency(self) -> float:
        """
        Misst die Systemlatenz.
        
        Returns:
            Latenz in Sekunden
        """
        # Simuliere Latenz-Messung
        return np.random.exponential(0.1)

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

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generiert einen Performance-Bericht.
        
        Returns:
            Dictionary mit Performance-Analyse
        """
        if not self.metrics:
            return {}

        report = {
            'duration': str(self.end_time - self.start_time),
            'metrics': {},
            'alerts': len(self.metrics['alerts'])
        }

        # Berechne Metriken-Statistiken
        for metric in ['cpu_usage', 'memory_usage', 'gpu_usage', 'disk_usage', 'latency']:
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

    def _analyze_alerts(self) -> Dict[str, Any]:
        """
        Analysiert aufgetretene Alerts.
        
        Returns:
            Dictionary mit Alert-Analyse
        """
        analysis = {
            'count_by_metric': {},
            'first_alert': None,
            'last_alert': None
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
                
        return analysis

    def get_recommendations(self) -> List[str]:
        """
        Generiert Optimierungsempfehlungen.
        
        Returns:
            Liste von Empfehlungen
        """
        report = self.get_performance_report()
        if not report:
            return []

        recommendations = []
        metrics = report['metrics']

        # CPU-Auslastung
        if metrics['cpu_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe CPU-Auslastung (P95: {metrics['cpu_usage']['p95']:.1f}%) - "
                "Prozesse optimieren oder Ressourcen erhöhen"
            )

        # Speicherauslastung
        if metrics['memory_usage']['p95'] > 80:
            recommendations.append(
                f"Hohe Speicherauslastung (P95: {metrics['memory_usage']['p95']:.1f}%) - "
                "Memory Leaks prüfen oder Speicher erhöhen"
            )

        # GPU-Auslastung
        if metrics['gpu_usage']['mean'] < 50:
            recommendations.append(
                f"Niedrige GPU-Auslastung (Durchschnitt: {metrics['gpu_usage']['mean']:.1f}%) - "
                "GPU-Nutzung optimieren"
            )

        # Latenz
        if metrics['latency']['p95'] > 1.0:
            recommendations.append(
                f"Hohe Latenz (P95: {metrics['latency']['p95']:.2f}s) - "
                "Performance-Bottlenecks identifizieren"
            )

        # Alert-Analyse
        if report.get('alert_analysis'):
            for metric, count in report['alert_analysis']['count_by_metric'].items():
                if count > 10:
                    recommendations.append(
                        f"Häufige Alerts für {metric} ({count} mal) - "
                        "Schwellenwerte überprüfen oder System optimieren"
                    )

        return recommendations

    def cleanup(self):
        """Räumt Monitoring-Ressourcen auf."""
        self.stop_monitoring()
        self.metrics = {} 