"""
Tests für den optimierten Datenloader.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from ml4t_project.data.optimized_loader import OptimizedDataLoader

@pytest.fixture
def loader():
    """Fixture für OptimizedDataLoader"""
    return OptimizedDataLoader(
        cache_dir="tests/cache",
        batch_size=100,
        use_gpu=torch.cuda.is_available()
    )

@pytest.fixture
def sample_data():
    """Fixture für Testdaten"""
    dates = pd.date_range(end=datetime.now(), periods=500)
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 10, 500),
        'high': np.random.normal(105, 10, 500),
        'low': np.random.normal(95, 10, 500),
        'close': np.random.normal(100, 10, 500),
        'volume': np.random.normal(1000000, 100000, 500)
    })

def test_loader_initialization(loader):
    """Test Loader-Initialisierung"""
    assert isinstance(loader, OptimizedDataLoader)
    assert loader.batch_size == 100
    assert loader.use_gpu == torch.cuda.is_available()

def test_data_loading(loader):
    """Test Datenladen"""
    symbols = ['AAPL', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = loader.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.gpu
def test_gpu_processing(loader, sample_data):
    """Test GPU-Verarbeitung"""
    if not torch.cuda.is_available():
        pytest.skip("GPU nicht verfügbar")
    
    processed_data = loader._process_on_gpu(sample_data)
    
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert any(col.endswith('_ma') for col in processed_data.columns)

def test_cpu_processing(loader, sample_data):
    """Test CPU-Verarbeitung"""
    processed_data = loader._process_on_cpu(sample_data)
    
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert any(col.endswith('_ma') for col in processed_data.columns)

def test_batch_processing(loader, sample_data):
    """Test Batch-Verarbeitung"""
    # Erstelle großen Datensatz
    large_data = pd.concat([sample_data] * 10)
    
    processed_data = loader._process_in_batches(large_data)
    
    assert len(processed_data) == len(large_data)
    assert any(col.endswith('_ma') for col in processed_data.columns)

def test_torch_tensor_conversion(loader, sample_data):
    """Test Tensor-Konvertierung"""
    tensor = loader.to_torch_tensor(sample_data)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == loader.device

@pytest.mark.gpu
def test_gpu_memory_cleanup(loader):
    """Test GPU-Speicherbereinigung"""
    if not torch.cuda.is_available():
        pytest.skip("GPU nicht verfügbar")
    
    # Erstelle dummy Tensor auf GPU
    dummy_tensor = torch.ones(1000, 1000).cuda()
    
    loader.clear_gpu_memory()
    
    # Prüfe ob Speicher freigegeben wurde
    assert torch.cuda.memory_allocated() == 0

@pytest.mark.integration
def test_full_pipeline_integration(loader):
    """Integrationstests für den Loader"""
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Lade und verarbeite Daten
    data = loader.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Validiere Ergebnisse
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert any(col.endswith('_ma') for col in data.columns)
    
    # Konvertiere zu Tensor
    tensor = loader.to_torch_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    
    # Cleanup
    if loader.use_gpu:
        loader.clear_gpu_memory()

@pytest.mark.benchmark
def test_performance_comparison(loader, sample_data):
    """Test Performance-Vergleich zwischen CPU und GPU"""
    import time
    
    # CPU-Verarbeitung
    cpu_start = time.time()
    _ = loader._process_on_cpu(sample_data)
    cpu_time = time.time() - cpu_start
    
    if loader.use_gpu:
        # GPU-Verarbeitung
        gpu_start = time.time()
        _ = loader._process_on_gpu(sample_data)
        gpu_time = time.time() - gpu_start
        
        # Vergleiche Zeiten
        assert gpu_time <= cpu_time, "GPU sollte schneller sein als CPU"
    
@pytest.fixture(autouse=True)
def cleanup(loader):
    """Cleanup nach Tests"""
    yield
    if loader.use_gpu:
        loader.clear_gpu_memory()

class TestOptimizedDataLoader:
    def setup_method(self):
        self.loader = OptimizedDataLoader(
            batch_size=1000,
            num_workers=4,
            prefetch_factor=2,
            cache_size=10000
        )
        
    def test_initialization(self):
        assert self.loader.is_initialized
        assert self.loader.batch_size == 1000
        assert self.loader.cache_size == 10000
        
    def test_data_loading(self):
        data = self.loader.load_data("test_data.csv")
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'timestamp' in data.columns
        
    def test_caching(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100),
            'value': range(100)
        })
        
        self.loader.cache_data("test_key", data)
        cached_data = self.loader.get_cached_data("test_key")
        assert cached_data.equals(data)
        
    def test_batch_processing(self):
        data = torch.randn(1000, 10)
        batches = self.loader.process_batches(data)
        assert len(batches) == 1
        assert batches[0].shape == (1000, 10)
        
    def test_preprocessing(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100),
            'value': range(100)
        })
        
        processed_data = self.loader.preprocess_data(data)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape[0] == 100
        
    def test_performance_metrics(self):
        metrics = self.loader.get_performance_metrics()
        assert 'loading_speed' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'memory_usage' in metrics
        
    def test_error_handling(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_data("non_existent_file.csv")
            
    def test_cleanup(self):
        self.loader.cleanup()
        assert not self.loader.is_initialized 