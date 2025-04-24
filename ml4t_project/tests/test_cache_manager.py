"""
Tests für den optimierten Cache-Manager.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from ..data.cache_manager import OptimizedCacheManager, CacheConfig
import torch
from ml4t_project.data.cache_manager import CacheManager

@pytest.fixture
def cache_manager():
    """Fixture für Cache-Manager."""
    cache_dir = "tests/cache"
    config = CacheConfig(
        max_size="1MB",
        expiry="1s",
        cleanup_interval="1s",
        max_items=10
    )
    manager = OptimizedCacheManager(cache_dir, config)
    yield manager
    # Cleanup
    manager.clear_cache()

def test_cache_initialization(cache_manager):
    """Test Cache-Initialisierung."""
    assert cache_manager.cache_dir.exists()
    assert isinstance(cache_manager.config, CacheConfig)
    assert cache_manager.config.max_size == "1MB"

def test_cache_data(cache_manager):
    """Test Daten-Caching."""
    # Erstelle Testdaten
    data = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100)
    })
    
    # Cache Daten
    cache_manager.cache_data("test_key", data)
    
    # Prüfe Cache-Existenz
    cache_file = cache_manager.cache_dir / "test_key.parquet"
    assert cache_file.exists()
    
    # Lade Daten zurück
    cached_data = cache_manager.get_cached_data("test_key")
    assert cached_data is not None
    pd.testing.assert_frame_equal(data, cached_data)

def test_cache_expiry(cache_manager):
    """Test Cache-Ablauf."""
    # Cache Daten
    data = pd.DataFrame({'A': [1, 2, 3]})
    cache_manager.cache_data("expiry_test", data)
    
    # Warte bis Cache abläuft
    time.sleep(2)
    
    # Prüfe, dass Daten nicht mehr im Cache sind
    cached_data = cache_manager.get_cached_data("expiry_test")
    assert cached_data is None

def test_cache_cleanup(cache_manager):
    """Test Cache-Bereinigung."""
    # Erstelle große Testdaten
    large_data = pd.DataFrame({
        'A': np.random.rand(10000),
        'B': np.random.rand(10000)
    })
    
    # Cache mehrere große Datensätze
    for i in range(5):
        cache_manager.cache_data(f"large_data_{i}", large_data)
    
    # Warte auf Cleanup
    time.sleep(2)
    
    # Prüfe Cache-Größe
    stats = cache_manager.get_cache_stats()
    assert stats['total_size'] <= cache_manager._convert_size("1MB")

def test_cache_stats(cache_manager):
    """Test Cache-Statistiken."""
    # Cache einige Daten
    for i in range(3):
        data = pd.DataFrame({'A': [i]})
        cache_manager.cache_data(f"stats_test_{i}", data)
    
    # Hole Statistiken
    stats = cache_manager.get_cache_stats()
    
    assert 'total_size' in stats
    assert 'num_items' in stats
    assert 'hit_rate' in stats
    assert 'last_cleanup' in stats
    assert stats['num_items'] == 3

def test_concurrent_cache_access(cache_manager):
    """Test gleichzeitiger Cache-Zugriff."""
    import threading
    
    def cache_access():
        try:
            data = pd.DataFrame({'A': [1]})
            cache_manager.cache_data("concurrent_test", data)
            cached_data = cache_manager.get_cached_data("concurrent_test")
            assert cached_data is not None
            return True
        except Exception:
            return False
    
    # Starte mehrere Threads
    threads = []
    results = []
    
    for _ in range(5):
        thread = threading.Thread(target=lambda: results.append(cache_access()))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Prüfe Ergebnisse
    assert all(results)

def test_cache_compression(cache_manager):
    """Test Cache-Kompression."""
    # Erstelle große Testdaten
    data = pd.DataFrame({
        'A': np.random.rand(10000),
        'B': np.random.rand(10000)
    })
    
    # Cache Daten
    cache_manager.cache_data("compression_test", data)
    
    # Prüfe Dateigröße
    cache_file = cache_manager.cache_dir / "compression_test.parquet"
    assert cache_file.stat().st_size < data.memory_usage().sum()

def test_cache_metadata(cache_manager):
    """Test Cache-Metadaten."""
    # Cache Daten
    data = pd.DataFrame({'A': [1]})
    cache_manager.cache_data("metadata_test", data)
    
    # Prüfe Metadaten
    metadata = cache_manager._cache_metadata.get("metadata_test")
    assert metadata is not None
    assert 'size' in metadata
    assert 'last_access' in metadata
    assert 'expiry_time' in metadata
    assert 'created_at' in metadata

def test_cache_clear(cache_manager):
    """Test Cache-Löschung."""
    # Cache Daten
    data = pd.DataFrame({'A': [1]})
    cache_manager.cache_data("clear_test", data)
    
    # Lösche Cache
    cache_manager.clear_cache("clear_test")
    
    # Prüfe, dass Daten gelöscht wurden
    assert not (cache_manager.cache_dir / "clear_test.parquet").exists()
    assert "clear_test" not in cache_manager._cache_metadata

def test_cache_error_handling(cache_manager):
    """Test Fehlerbehandlung."""
    # Versuche ungültige Daten zu cachen
    cache_manager.cache_data("error_test", None)
    
    # Prüfe, dass keine ungültigen Daten gecacht wurden
    assert not (cache_manager.cache_dir / "error_test.parquet").exists()
    
    # Versuche ungültige Daten zu laden
    cached_data = cache_manager.get_cached_data("nonexistent_key")
    assert cached_data is None

class TestCacheManager:
    def setup_method(self):
        self.cache = CacheManager(
            max_size=1000,
            compression=True,
            cleanup_interval=timedelta(minutes=5)
        )
        
    def test_initialization(self):
        assert self.cache.is_initialized
        assert self.cache.max_size == 1000
        assert self.cache.compression == True
        
    def test_data_caching(self):
        key = "test_data"
        data = torch.randn(100, 10)
        metadata = {
            'timestamp': datetime.now(),
            'source': 'test'
        }
        
        self.cache.set(key, data, metadata)
        cached_data, cached_metadata = self.cache.get(key)
        
        assert torch.allclose(data, cached_data)
        assert metadata['source'] == cached_metadata['source']
        
    def test_cache_eviction(self):
        # Fülle Cache bis zum Maximum
        for i in range(1001):
            self.cache.set(f"key_{i}", torch.randn(10), {})
            
        # Überprüfe, ob älteste Einträge entfernt wurden
        assert len(self.cache) <= 1000
        
    def test_compression(self):
        key = "compressed_data"
        data = torch.randn(1000, 1000)
        
        self.cache.set(key, data, {})
        cached_data, _ = self.cache.get(key)
        
        assert torch.allclose(data, cached_data)
        
    def test_metadata_tracking(self):
        key = "test_metadata"
        metadata = {
            'timestamp': datetime.now(),
            'source': 'test',
            'size': 1000
        }
        
        self.cache.set(key, torch.randn(10), metadata)
        _, cached_metadata = self.cache.get(key)
        
        assert metadata['source'] == cached_metadata['source']
        assert metadata['size'] == cached_metadata['size']
        
    def test_performance_metrics(self):
        metrics = self.cache.get_performance_metrics()
        assert 'hit_rate' in metrics
        assert 'memory_usage' in metrics
        assert 'latency' in metrics
        
    def test_error_handling(self):
        with pytest.raises(KeyError):
            self.cache.get("non_existent_key")
            
    def test_cleanup(self):
        self.cache.cleanup()
        assert len(self.cache) == 0 