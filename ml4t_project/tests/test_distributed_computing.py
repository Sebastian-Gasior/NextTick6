import pytest
import torch
from ml4t_project.computing.distributed_computer import DistributedComputer
from datetime import datetime

class TestDistributedComputer:
    def setup_method(self):
        self.computer = DistributedComputer(world_size=2)
        
    def test_initialization(self):
        assert self.computer.is_initialized
        assert self.computer.world_size == 2
        
    def test_data_distribution(self):
        data = torch.randn(100, 10)
        chunks = self.computer.distribute_data(data)
        assert len(chunks) == 2
        assert chunks[0].shape[0] == 50
        
    def test_model_distribution(self):
        model = torch.nn.Linear(10, 1)
        distributed_model = self.computer.distribute_model(model)
        assert isinstance(distributed_model, torch.nn.parallel.DistributedDataParallel)
        
    def test_batch_processing(self):
        data = torch.randn(32, 10)
        model = torch.nn.Linear(10, 1)
        output = self.computer.process_batch(data, model)
        assert output.shape == (32, 1)
        
    def test_performance_metrics(self):
        metrics = self.computer.get_performance_metrics()
        assert 'throughput' in metrics
        assert 'latency' in metrics
        assert 'gpu_utilization' in metrics
        
    def test_resource_monitoring(self):
        resources = self.computer.monitor_resources()
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'gpu_usage' in resources
        
    def test_network_failure_simulation(self):
        with pytest.raises(RuntimeError):
            self.computer.simulate_network_failure()
            
    def test_cleanup(self):
        self.computer.cleanup()
        assert not self.computer.is_initialized 