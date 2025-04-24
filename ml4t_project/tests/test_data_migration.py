"""
Tests für die Datenpipeline mit echten Daten.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from ..data.data_migration import DataMigrationConfig, DataValidator, HybridDataPipeline
import torch
from ml4t_project.data.data_migration import DataMigrator

@pytest.fixture
def config():
    """Fixture für Test-Konfiguration"""
    return DataMigrationConfig("tests/test_config.yaml")

@pytest.fixture
def validator():
    """Fixture für DataValidator"""
    return DataValidator()

@pytest.fixture
def pipeline():
    """Fixture für HybridDataPipeline"""
    return HybridDataPipeline("tests/test_config.yaml")

@pytest.fixture
def sample_size():
    """Fixture für Testdatengröße"""
    return 1000

def test_config_loading(config):
    """Test Konfigurationsladung"""
    assert config.config is not None
    assert config.config['migration']['use_real_data'] is True
    assert config.config['migration']['fallback_to_dummy'] is False
    assert len(config.config['data']['symbols']) > 0

def test_data_structure_validation(validator, pipeline, sample_size):
    """Test Datenstrukturvalidierung"""
    data = pipeline.get_data(sample_size)
    assert validator.validate_structure(data) is True
    
    # Prüfe erforderliche Spalten
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in data.columns for col in required_columns)

def test_data_quality_validation(validator, pipeline, sample_size):
    """Test Datenqualitätsvalidierung"""
    data = pipeline.get_data(sample_size)
    is_valid, metrics = validator.validate_data_quality(data)
    
    assert is_valid is True
    assert metrics['missing_values'] == 0
    assert metrics['duplicates'] == 0
    assert metrics['invalid_values'] == 0

def test_real_data_loading(pipeline, sample_size):
    """Test Laden echter Daten"""
    data = pipeline.get_data(sample_size)
    
    assert len(data) == sample_size
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_monotonic_increasing
    assert not data.empty
    
    # Prüfe Datenqualität
    assert not data.isnull().any().any()
    assert all(data['volume'] >= 0)
    assert all(data['low'] <= data['high'])

@pytest.mark.performance
def test_performance_validation(pipeline, sample_size):
    """Test Performance-Validierung"""
    data = pipeline.get_data(sample_size)
    is_valid, metrics = pipeline.validate_performance(data)
    
    assert is_valid is True
    assert metrics['processing_time_ms'] <= pipeline.config.config['performance']['max_processing_time_ms']
    assert metrics['memory_usage_mb'] <= pipeline.config.config['performance']['max_memory_usage_mb']

@pytest.mark.integration
def test_full_pipeline_integration(pipeline):
    """Integrationstests für die Pipeline"""
    sizes = [100, 500, 1000]
    
    for size in sizes:
        data = pipeline.get_data(size)
        
        # Validiere Größe
        assert len(data) == size
        
        # Validiere Struktur
        assert pipeline.validator.validate_structure(data)
        
        # Validiere Qualität
        is_valid, metrics = pipeline.validator.validate_data_quality(data)
        assert is_valid
        
        # Validiere Performance
        is_valid, metrics = pipeline.validate_performance(data)
        assert is_valid

@pytest.mark.error
def test_error_handling(pipeline):
    """Test Fehlerbehandlung"""
    # Test mit ungültiger Größe
    with pytest.raises(ValueError):
        pipeline.get_data(-1)
    
    with pytest.raises(ValueError):
        pipeline.get_data(0)

def test_data_consistency(pipeline):
    """Test Datenkonsistenz"""
    # Lade gleiche Daten mehrmals
    data1 = pipeline.get_data(100)
    data2 = pipeline.get_data(100)
    
    # Vergleiche Statistiken
    stats1 = data1.describe()
    stats2 = data2.describe()
    
    pd.testing.assert_frame_equal(
        stats1,
        stats2,
        rtol=0.1  # 10% Toleranz für Echtzeitdaten
    )

@pytest.mark.performance
def test_memory_management(pipeline):
    """Test Speichermanagement"""
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Lade große Datenmenge
    data = pipeline.get_data(5000)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    # Speicherverbrauch sollte kontrolliert sein
    assert memory_increase < 1024  # Max 1GB Zunahme
    
    # Cleanup
    del data
    import gc
    gc.collect()

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup nach Tests"""
    yield
    # Lösche Test-Konfiguration und Cache
    for path in ['tests/test_config.yaml', 'tests/cache']:
        try:
            if Path(path).is_file():
                Path(path).unlink()
            elif Path(path).is_dir():
                import shutil
                shutil.rmtree(path)
        except:
            pass

class TestDataMigrator:
    def setup_method(self):
        self.migrator = DataMigrator(
            source_path="data/source",
            target_path="data/target",
            batch_size=1000,
            compression=True
        )
        
    def test_initialization(self):
        assert self.migrator.is_initialized
        assert self.migrator.batch_size == 1000
        assert self.migrator.compression == True
        
    def test_data_validation(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100),
            'value': range(100)
        })
        
        is_valid = self.migrator.validate_data(data)
        assert is_valid
        
    def test_batch_processing(self):
        data = torch.randn(1000, 10)
        processed_data = self.migrator.process_batch(data)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape == (1000, 10)
        
    def test_compression(self):
        data = torch.randn(1000, 1000)
        compressed_data = self.migrator.compress_data(data)
        assert isinstance(compressed_data, bytes)
        
    def test_decompression(self):
        data = torch.randn(1000, 1000)
        compressed_data = self.migrator.compress_data(data)
        decompressed_data = self.migrator.decompress_data(compressed_data)
        assert torch.allclose(data, decompressed_data)
        
    def test_metadata_tracking(self):
        metadata = {
            'source': 'test',
            'timestamp': datetime.now(),
            'size': 1000
        }
        
        self.migrator.track_metadata(metadata)
        tracked_metadata = self.migrator.get_metadata()
        assert metadata['source'] == tracked_metadata['source']
        
    def test_error_handling(self):
        with pytest.raises(RuntimeError):
            self.migrator.simulate_migration_error()
            
    def test_performance_metrics(self):
        metrics = self.migrator.get_performance_metrics()
        assert 'processing_speed' in metrics
        assert 'compression_ratio' in metrics
        assert 'error_rate' in metrics
        
    def test_cleanup(self):
        self.migrator.cleanup()
        assert not self.migrator.is_initialized 