"""
Stabilitäts-Tester für das ML4T-Projekt.
"""
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np

class StabilityTester:
    def __init__(self,
                 test_duration: timedelta = timedelta(hours=24),
                 check_interval: timedelta = timedelta(minutes=5),
                 error_threshold: float = 0.01):
        """
        Initialisiert den Stabilitäts-Tester.
        
        Args:
            test_duration: Gesamtdauer des Tests
            check_interval: Intervall für Stabilitätsprüfungen
            error_threshold: Maximale erlaubte Fehlerrate
        """
        self.test_duration = test_duration
        self.check_interval = check_interval
        self.error_threshold = error_threshold
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)

    def start_test(self):
        """Startet den Stabilitätstest."""
        self.start_time = datetime.now()
        self.is_running = True
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'error_rate': [],
            'response_time': []
        }
        self.logger.info("Stabilitätstest gestartet")

    def stop_test(self):
        """Stoppt den Stabilitätstest."""
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("Stabilitätstest beendet")

    def check_stability(self) -> Dict[str, Any]:
        """
        Führt eine Stabilitätsprüfung durch.
        
        Returns:
            Dictionary mit Stabilitätsmetriken
        """
        if not self.is_running:
            raise RuntimeError("Test nicht aktiv")

        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'error_rate': self._calculate_error_rate(),
            'response_time': self._measure_response_time()
        }

        # Aktualisiere Metriken
        for key, value in metrics.items():
            self.metrics[key].append(value)

        return metrics

    def _calculate_error_rate(self) -> float:
        """Berechnet die aktuelle Fehlerrate."""
        # Simuliere Fehlerrate
        return np.random.beta(1, 100)  # Meist niedrige Werte

    def _measure_response_time(self) -> float:
        """Misst die Antwortzeit."""
        # Simuliere Antwortzeit
        return np.random.exponential(0.1)  # Meist schnelle Antworten

    def get_stability_report(self) -> Dict[str, Any]:
        """
        Generiert einen Stabilitätsbericht.
        
        Returns:
            Dictionary mit Stabilitätsanalyse
        """
        if not self.metrics:
            return {}

        report = {
            'duration': str(self.end_time - self.start_time),
            'metrics': {}
        }

        for metric, values in self.metrics.items():
            values = np.array(values)
            report['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'stability_score': self._calculate_stability_score(values)
            }

        report['overall_stability'] = self._calculate_overall_stability()
        report['recommendations'] = self._generate_recommendations()

        return report

    def _calculate_stability_score(self, values: np.ndarray) -> float:
        """
        Berechnet einen Stabilitätsscore für eine Metrik.
        
        Args:
            values: Array von Metrikwerten
            
        Returns:
            Stabilitätsscore zwischen 0 und 1
        """
        if len(values) < 2:
            return 1.0

        # Berechne Variationskoeffizient
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
        
        # Transformiere zu Score zwischen 0 und 1
        score = 1 / (1 + cv)
        return float(score)

    def _calculate_overall_stability(self) -> float:
        """
        Berechnet die Gesamtstabilität.
        
        Returns:
            Stabilitätsscore zwischen 0 und 1
        """
        scores = []
        for values in self.metrics.values():
            scores.append(self._calculate_stability_score(np.array(values)))
        return float(np.mean(scores)) if scores else 0.0

    def _generate_recommendations(self) -> list:
        """
        Generiert Verbesserungsempfehlungen.
        
        Returns:
            Liste von Empfehlungen
        """
        recommendations = []
        metrics = self.get_stability_report()['metrics']

        # CPU-Auslastung
        if metrics['cpu_usage']['mean'] > 80:
            recommendations.append("CPU-Auslastung reduzieren")

        # Speicherverbrauch
        if metrics['memory_usage']['mean'] > 1000:  # MB
            recommendations.append("Speicherverbrauch optimieren")

        # Fehlerrate
        if metrics['error_rate']['mean'] > self.error_threshold:
            recommendations.append("Fehlerbehandlung verbessern")

        # Antwortzeit
        if metrics['response_time']['mean'] > 1.0:  # Sekunden
            recommendations.append("Antwortzeit optimieren")

        return recommendations

    def simulate_load(self, duration: Optional[timedelta] = None):
        """
        Simuliert Last für Testzwecke.
        
        Args:
            duration: Optionale Testdauer
        """
        if duration is None:
            duration = timedelta(minutes=1)

        start_time = datetime.now()
        end_time = start_time + duration

        while datetime.now() < end_time:
            # Simuliere CPU-Last
            _ = [i * i for i in range(1000)]
            
            # Simuliere Speicherverbrauch
            _ = [0] * 1000000
            
            time.sleep(0.1)  # Kleine Pause 