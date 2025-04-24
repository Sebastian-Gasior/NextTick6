"""
Migrations-Stress-Tester für das ML4T-Projekt.
"""
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class MigrationStressTester:
    def __init__(self,
                 data_size: int = 1000000,
                 chunk_size: int = 10000,
                 num_workers: int = 4,
                 test_duration: timedelta = timedelta(minutes=30)):
        """
        Initialisiert den Migrations-Stress-Tester.
        
        Args:
            data_size: Gesamtgröße der zu migrierenden Daten
            chunk_size: Größe der Daten-Chunks
            num_workers: Anzahl paralleler Worker
            test_duration: Maximale Testdauer
        """
        self.data_size = data_size
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.test_duration = test_duration
        self.metrics = {}
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()

    def start_test(self):
        """Startet den Migrations-Stress-Test."""
        self.start_time = datetime.now()
        self.is_running = True
        self._stop_event.clear()
        self.metrics = {
            'migration_times': [],
            'error_count': 0,
            'chunks_processed': 0,
            'cpu_usage': [],
            'memory_usage': []
        }
        self.logger.info("Migrations-Stress-Test gestartet")
        
        # Starte Monitoring
        self._start_monitoring()
        
        # Starte Migration
        self._run_migration()

    def stop_test(self):
        """Stoppt den Migrations-Stress-Test."""
        self._stop_event.set()
        self.end_time = datetime.now()
        self.is_running = False
        self.logger.info("Migrations-Stress-Test beendet")

    def _start_monitoring(self):
        """Startet das Ressourcen-Monitoring."""
        def monitor():
            while not self._stop_event.is_set():
                self.metrics['cpu_usage'].append(psutil.cpu_percent())
                self.metrics['memory_usage'].append(
                    psutil.Process().memory_info().rss / 1024 / 1024
                )
                time.sleep(1)

        threading.Thread(target=monitor, daemon=True).start()

    def _run_migration(self):
        """Führt die Datenmigration durch."""
        num_chunks = self.data_size // self.chunk_size
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for chunk_id in range(num_chunks):
                if self._stop_event.is_set():
                    break
                    
                future = executor.submit(
                    self._process_chunk,
                    chunk_id,
                    self.chunk_size
                )
                futures.append(future)
                
            # Warte auf Ergebnisse
            for future in futures:
                try:
                    migration_time = future.result()
                    if migration_time is not None:
                        self.metrics['migration_times'].append(migration_time)
                        self.metrics['chunks_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Fehler bei Chunk-Verarbeitung: {e}")
                    self.metrics['error_count'] += 1

    def _process_chunk(self, chunk_id: int, size: int) -> Optional[float]:
        """
        Verarbeitet einen Daten-Chunk.
        
        Args:
            chunk_id: ID des Chunks
            size: Größe des Chunks
            
        Returns:
            Verarbeitungszeit in Sekunden oder None bei Fehler
        """
        try:
            start_time = time.time()
            
            # Simuliere Datenverarbeitung
            self._simulate_processing(size)
            
            # Simuliere gelegentliche Fehler
            if np.random.random() < 0.01:  # 1% Fehlerrate
                raise Exception(f"Fehler bei Chunk {chunk_id}")
            
            return time.time() - start_time
        except Exception as e:
            self.logger.error(f"Fehler bei Chunk {chunk_id}: {e}")
            return None

    def _simulate_processing(self, size: int):
        """
        Simuliert Datenverarbeitung.
        
        Args:
            size: Datengröße
        """
        # Simuliere CPU-Last
        _ = [i * i for i in range(size // 1000)]
        
        # Simuliere I/O
        time.sleep(np.random.exponential(0.01))
        
        # Simuliere Speicherverbrauch
        _ = [0] * (size // 100)

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
            'data_size': self.data_size,
            'chunk_size': self.chunk_size,
            'num_workers': self.num_workers,
            'metrics': {}
        }

        # Berechne Metriken
        migration_times = np.array(self.metrics['migration_times'])
        total_time = (self.end_time - self.start_time).total_seconds()

        report['metrics']['migration_time'] = {
            'mean': float(np.mean(migration_times)),
            'p50': float(np.percentile(migration_times, 50)),
            'p95': float(np.percentile(migration_times, 95)),
            'total': float(np.sum(migration_times))
        }

        report['metrics']['throughput'] = (
            self.metrics['chunks_processed'] * self.chunk_size / total_time
        )
        
        report['metrics']['error_rate'] = (
            self.metrics['error_count'] /
            (self.metrics['chunks_processed'] + self.metrics['error_count'])
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

        # Migration Time
        if metrics['migration_time']['p95'] > 1.0:
            recommendations.append(
                f"Chunk-Verarbeitungszeit optimieren "
                f"(P95: {metrics['migration_time']['p95']:.2f}s)"
            )

        # Throughput
        expected_throughput = self.data_size / self.test_duration.total_seconds()
        if metrics['throughput'] < expected_throughput * 0.8:
            recommendations.append(
                f"Durchsatz verbessern "
                f"(Erreicht: {metrics['throughput']:.0f} items/s)"
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

        # Chunk Size
        if metrics['migration_time']['mean'] > 0.5:
            recommendations.append(
                "Chunk-Größe optimieren für bessere Performance"
            )

        # Worker Count
        if metrics['cpu_usage']['mean'] < 50:
            recommendations.append(
                "Anzahl Worker erhöhen für bessere Parallelisierung"
            )

        return recommendations 