"""
Last-Tester für das ML4T-Projekt.
"""
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class LoadTester:
    def __init__(self,
                 target_rps: float = 100.0,
                 test_duration: timedelta = timedelta(minutes=10),
                 warmup_duration: timedelta = timedelta(minutes=1),
                 cooldown_duration: timedelta = timedelta(minutes=1)):
        """
        Initialisiert den Last-Tester.
        
        Args:
            target_rps: Ziel-Requests pro Sekunde
            test_duration: Dauer des Tests
            warmup_duration: Dauer der Aufwärmphase
            cooldown_duration: Dauer der Abkühlphase
        """
        self.target_rps = target_rps
        self.test_duration = test_duration
        self.warmup_duration = warmup_duration
        self.cooldown_duration = cooldown_duration
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)

    def start_test(self):
        """Startet den Last-Test."""
        self.start_time = datetime.now()
        self.is_running = True
        self.metrics = {
            'response_times': [],
            'error_count': 0,
            'request_count': 0,
            'cpu_usage': [],
            'memory_usage': []
        }
        self.logger.info("Last-Test gestartet")

        # Führe Test-Phasen aus
        self._run_warmup()
        self._run_test()
        self._run_cooldown()

    def stop_test(self):
        """Stoppt den Last-Test."""
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("Last-Test beendet")

    def _run_warmup(self):
        """Führt die Aufwärmphase aus."""
        self.logger.info("Starte Aufwärmphase")
        self._run_phase(self.warmup_duration, self.target_rps * 0.5)

    def _run_test(self):
        """Führt die Testphase aus."""
        self.logger.info("Starte Testphase")
        self._run_phase(self.test_duration, self.target_rps)

    def _run_cooldown(self):
        """Führt die Abkühlphase aus."""
        self.logger.info("Starte Abkühlphase")
        self._run_phase(self.cooldown_duration, self.target_rps * 0.5)

    def _run_phase(self, duration: timedelta, target_rps: float):
        """
        Führt eine Testphase aus.
        
        Args:
            duration: Dauer der Phase
            target_rps: Ziel-Requests pro Sekunde
        """
        start_time = datetime.now()
        end_time = start_time + duration

        with ThreadPoolExecutor() as executor:
            while datetime.now() < end_time and self.is_running:
                # Berechne Anzahl paralleler Requests
                parallel_requests = int(target_rps / 10)  # 10 Batches pro Sekunde
                
                # Starte Requests parallel
                futures = []
                for _ in range(parallel_requests):
                    future = executor.submit(self._send_request)
                    futures.append(future)
                
                # Sammle Ergebnisse
                for future in futures:
                    try:
                        response_time, success = future.result()
                        self.metrics['response_times'].append(response_time)
                        self.metrics['request_count'] += 1
                        if not success:
                            self.metrics['error_count'] += 1
                    except Exception as e:
                        self.logger.error(f"Fehler bei Request: {e}")
                        self.metrics['error_count'] += 1
                
                # Erfasse System-Metriken
                self.metrics['cpu_usage'].append(psutil.cpu_percent())
                self.metrics['memory_usage'].append(
                    psutil.Process().memory_info().rss / 1024 / 1024
                )
                
                # Warte bis zum nächsten Batch
                time.sleep(0.1)  # 10 Batches pro Sekunde

    def _send_request(self) -> tuple:
        """
        Simuliert einen Request.
        
        Returns:
            Tuple aus (response_time, success)
        """
        start_time = time.time()
        
        try:
            # Simuliere Verarbeitung
            processing_time = np.random.exponential(0.1)
            time.sleep(processing_time)
            
            # Simuliere gelegentliche Fehler
            if np.random.random() < 0.01:  # 1% Fehlerrate
                raise Exception("Simulierter Fehler")
            
            response_time = time.time() - start_time
            return response_time, True
        except Exception:
            response_time = time.time() - start_time
            return response_time, False

    def get_test_report(self) -> Dict[str, Any]:
        """
        Generiert einen Testbericht.
        
        Returns:
            Dictionary mit Testergebnissen
        """
        if not self.metrics:
            return {}

        report = {
            'duration': str(self.end_time - self.start_time),
            'target_rps': self.target_rps,
            'metrics': {}
        }

        # Berechne Metriken
        response_times = np.array(self.metrics['response_times'])
        total_time = (self.end_time - self.start_time).total_seconds()

        report['metrics']['response_time'] = {
            'mean': float(np.mean(response_times)),
            'p50': float(np.percentile(response_times, 50)),
            'p95': float(np.percentile(response_times, 95)),
            'p99': float(np.percentile(response_times, 99))
        }

        report['metrics']['throughput'] = self.metrics['request_count'] / total_time
        report['metrics']['error_rate'] = (
            self.metrics['error_count'] / self.metrics['request_count']
            if self.metrics['request_count'] > 0 else 0
        )

        report['metrics']['cpu_usage'] = {
            'mean': float(np.mean(self.metrics['cpu_usage'])),
            'max': float(np.max(self.metrics['cpu_usage']))
        }

        report['metrics']['memory_usage'] = {
            'mean': float(np.mean(self.metrics['memory_usage'])),
            'max': float(np.max(self.metrics['memory_usage']))
        }

        return report

    def get_recommendations(self) -> List[str]:
        """
        Generiert Optimierungsempfehlungen.
        
        Returns:
            Liste von Empfehlungen
        """
        report = self.get_test_report()
        if not report:
            return []

        recommendations = []
        metrics = report['metrics']

        # Throughput
        achieved_rps = metrics['throughput']
        if achieved_rps < self.target_rps * 0.9:
            recommendations.append(
                f"Durchsatz verbessern (Ziel: {self.target_rps:.1f} RPS, "
                f"Erreicht: {achieved_rps:.1f} RPS)"
            )

        # Response Time
        if metrics['response_time']['p95'] > 1.0:
            recommendations.append(
                f"Response Time optimieren "
                f"(P95: {metrics['response_time']['p95']:.2f}s)"
            )

        # Error Rate
        if metrics['error_rate'] > 0.01:
            recommendations.append(
                f"Fehlerrate reduzieren "
                f"({metrics['error_rate']*100:.1f}%)"
            )

        # Resource Usage
        if metrics['cpu_usage']['mean'] > 80:
            recommendations.append(
                f"CPU-Auslastung optimieren "
                f"(Durchschnitt: {metrics['cpu_usage']['mean']:.1f}%)"
            )

        if metrics['memory_usage']['max'] > 1000:
            recommendations.append(
                f"Speicherverbrauch optimieren "
                f"(Maximum: {metrics['memory_usage']['max']:.0f} MB)"
            )

        return recommendations 