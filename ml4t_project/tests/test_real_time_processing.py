import pytest
import torch
from ml4t_project.processing.real_time_processor import RealTimeProcessor
from datetime import datetime

class TestRealTimeProcessor:
    def setup_method(self):
        self.processor = RealTimeProcessor(buffer_size=100, processing_interval=0.1)
        
    def test_initialization(self):
        assert self.processor.is_initialized
        assert self.processor.buffer_size == 100
        assert self.processor.processing_interval == 0.1
        
    def test_data_ingestion(self):
        data_point = {
            'timestamp': datetime.now(),
            'data': torch.randn(10)
        }
        self.processor.ingest_data(data_point)
        assert len(self.processor.data_buffer) == 1
        
    def test_invalid_data_ingestion(self):
        with pytest.raises(ValueError):
            self.processor.ingest_data({
                'timestamp': 'invalid',
                'data': torch.randn(10)
            })
            
        with pytest.raises(ValueError):
            self.processor.ingest_data({
                'timestamp': datetime.now(),
                'data': 'invalid'
            })
            
    def test_data_processing(self):
        data_point = {
            'timestamp': datetime.now(),
            'data': torch.randn(10)
        }
        processed_data = self.processor.process_data(data_point)
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape == (10,)
        
    def test_prediction(self):
        data_point = {
            'timestamp': datetime.now(),
            'data': torch.randn(10)
        }
        prediction = self.processor.predict(data_point)
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == (10,)
        assert torch.all(prediction >= 0) and torch.all(prediction <= 1)
        
    def test_resource_monitoring(self):
        resources = self.processor.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'processing_rate' in resources
        assert 'latency' in resources
        
    def test_processing_error_simulation(self):
        with pytest.raises(RuntimeError):
            self.processor.simulate_processing_error()
            
    def test_cleanup(self):
        self.processor.cleanup()
        assert not self.processor.is_initialized
        assert len(self.processor.data_buffer) == 0 