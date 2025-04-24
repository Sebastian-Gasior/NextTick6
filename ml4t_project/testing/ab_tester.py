"""
A/B-Tester für das ML4T-Projekt.
"""
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ABTester:
    def __init__(self,
                 variant_a: Callable,
                 variant_b: Callable,
                 sample_size: int = 1000,
                 confidence_level: float = 0.95,
                 test_duration: timedelta = timedelta(hours=24)):
        """
        Initialisiert den A/B-Tester.
        
        Args:
            variant_a: Funktion für Variante A
            variant_b: Funktion für Variante B
            sample_size: Stichprobengröße pro Variante
            confidence_level: Konfidenzniveau
            test_duration: Maximale Testdauer
        """
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.test_duration = test_duration
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()

    def start_test(self):
        """Startet den A/B-Test."""
        self.start_time = datetime.now()
        self.is_running = True
        self._stop_event.clear()
        self.metrics = {
            'variant_a': {
                'response_times': [],
                'success_count': 0,
                'error_count': 0
            },
            'variant_b': {
                'response_times': [],
                'success_count': 0,
                'error_count': 0
            }
        }
        self.logger.info("A/B-Test gestartet")
        
        # Starte Test
        self._run_test()

    def stop_test(self):
        """Stoppt den A/B-Test."""
        self._stop_event.set()
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("A/B-Test beendet")

    def _run_test(self):
        """Führt den A/B-Test durch."""
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # Teste beide Varianten parallel
            for _ in range(self.sample_size):
                if self._stop_event.is_set():
                    break
                    
                # Teste Variante A
                future_a = executor.submit(self._test_variant, 'a')
                futures.append(future_a)
                
                # Teste Variante B
                future_b = executor.submit(self._test_variant, 'b')
                futures.append(future_b)
            
            # Sammle Ergebnisse
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Fehler im Test: {e}")

    def _test_variant(self, variant: str):
        """
        Testet eine Variante.
        
        Args:
            variant: 'a' oder 'b'
        """
        try:
            # Wähle Variante
            func = self.variant_a if variant == 'a' else self.variant_b
            metrics_key = 'variant_a' if variant == 'a' else 'variant_b'
            
            # Führe Test durch
            start_time = time.time()
            success = func()
            response_time = time.time() - start_time
            
            # Erfasse Metriken
            self.metrics[metrics_key]['response_times'].append(response_time)
            if success:
                self.metrics[metrics_key]['success_count'] += 1
            else:
                self.metrics[metrics_key]['error_count'] += 1
                
        except Exception as e:
            self.logger.error(f"Fehler in Variante {variant}: {e}")
            self.metrics[metrics_key]['error_count'] += 1

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
            'sample_size': self.sample_size,
            'confidence_level': self.confidence_level,
            'metrics': {}
        }

        # Berechne Metriken für beide Varianten
        for variant in ['variant_a', 'variant_b']:
            metrics = self.metrics[variant]
            response_times = np.array(metrics['response_times'])
            total_requests = (
                metrics['success_count'] + metrics['error_count']
            )

            report['metrics'][variant] = {
                'response_time': {
                    'mean': float(np.mean(response_times)),
                    'p50': float(np.percentile(response_times, 50)),
                    'p95': float(np.percentile(response_times, 95))
                },
                'success_rate': (
                    metrics['success_count'] / total_requests
                    if total_requests > 0 else 0
                ),
                'error_rate': (
                    metrics['error_count'] / total_requests
                    if total_requests > 0 else 0
                ),
                'sample_size': total_requests
            }

        # Führe statistische Tests durch
        report['analysis'] = self._analyze_results()

        return report

    def _analyze_results(self) -> Dict[str, Any]:
        """
        Führt statistische Analyse durch.
        
        Returns:
            Dictionary mit Analyseergebnissen
        """
        analysis = {}
        
        # Response Time Analyse
        times_a = np.array(self.metrics['variant_a']['response_times'])
        times_b = np.array(self.metrics['variant_b']['response_times'])
        
        # T-Test für Response Times
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(times_a, times_b)
        
        analysis['response_time'] = {
            'difference': float(np.mean(times_b) - np.mean(times_a)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < (1 - self.confidence_level)
        }
        
        # Success Rate Analyse
        success_rate_a = (
            self.metrics['variant_a']['success_count'] /
            (self.metrics['variant_a']['success_count'] +
             self.metrics['variant_a']['error_count'])
        )
        success_rate_b = (
            self.metrics['variant_b']['success_count'] /
            (self.metrics['variant_b']['success_count'] +
             self.metrics['variant_b']['error_count'])
        )
        
        analysis['success_rate'] = {
            'difference': float(success_rate_b - success_rate_a),
            'significant': abs(success_rate_b - success_rate_a) > 0.1
        }
        
        return analysis

    def get_recommendations(self) -> List[str]:
        """
        Generiert Empfehlungen basierend auf den Testergebnissen.
        
        Returns:
            Liste von Empfehlungen
        """
        report = self.get_test_report()
        if not report:
            return []

        recommendations = []
        metrics = report['metrics']
        analysis = report['analysis']

        # Response Time Analyse
        if analysis['response_time']['significant']:
            faster_variant = (
                'B' if analysis['response_time']['difference'] < 0 else 'A'
            )
            recommendations.append(
                f"Variante {faster_variant} ist signifikant schneller "
                f"(Differenz: {abs(analysis['response_time']['difference']):.3f}s)"
            )

        # Success Rate Analyse
        if analysis['success_rate']['significant']:
            better_variant = (
                'B' if analysis['success_rate']['difference'] > 0 else 'A'
            )
            recommendations.append(
                f"Variante {better_variant} hat eine signifikant höhere Erfolgsrate "
                f"(Differenz: {abs(analysis['success_rate']['difference'])*100:.1f}%)"
            )

        # Performance Empfehlungen
        for variant in ['variant_a', 'variant_b']:
            if metrics[variant]['response_time']['p95'] > 1.0:
                variant_name = 'A' if variant == 'variant_a' else 'B'
                recommendations.append(
                    f"Performance von Variante {variant_name} verbessern "
                    f"(P95: {metrics[variant]['response_time']['p95']:.2f}s)"
                )

        # Error Rate Empfehlungen
        for variant in ['variant_a', 'variant_b']:
            if metrics[variant]['error_rate'] > 0.01:
                variant_name = 'A' if variant == 'variant_a' else 'B'
                recommendations.append(
                    f"Fehlerrate von Variante {variant_name} reduzieren "
                    f"({metrics[variant]['error_rate']*100:.1f}%)"
                )

        return recommendations 