import pytest
import torch
from ml4t_project.optimization.gpu_optimizer import GPUOptimizer

class TestGPUOptimization:
    def setup_method(self):
        self.optimizer = GPUOptimizer()
        
    def test_memory_management(self):
        # Teste Speicherverwaltung
        initial_memory = torch.cuda.memory_allocated()
        tensor = torch.randn(1000, 1000, device='cuda')
        self.optimizer.optimize_memory(tensor)
        final_memory = torch.cuda.memory_allocated()
        assert final_memory < initial_memory + 1000000  # Maximal 1MB mehr
        
    def test_mixed_precision(self):
        # Teste Mixed Precision Training
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        loss = self.optimizer.train_with_mixed_precision(model, optimizer)
        assert isinstance(loss, float)
        
    def test_tensor_optimization(self):
        # Teste Tensor-Optimierung
        tensor = torch.randn(1000, 1000, device='cuda')
        optimized_tensor = self.optimizer.optimize_tensor(tensor)
        assert optimized_tensor.is_contiguous()
        assert optimized_tensor.device.type == 'cuda'
        
    def test_parallel_processing(self):
        # Teste parallele Verarbeitung
        data = [torch.randn(100, 100) for _ in range(4)]
        results = self.optimizer.process_in_parallel(data)
        assert len(results) == 4
        assert all(isinstance(r, torch.Tensor) for r in results)
        
    def test_performance_metrics(self):
        # Teste Performance-Metriken
        metrics = self.optimizer.get_performance_metrics()
        assert 'gpu_utilization' in metrics
        assert 'memory_usage' in metrics
        assert 'latency' in metrics
        assert 'throughput' in metrics
        
    def test_error_handling(self):
        # Teste Fehlerbehandlung
        with pytest.raises(RuntimeError):
            self.optimizer.simulate_gpu_error()
            
    def test_cleanup(self):
        # Teste Bereinigung
        self.optimizer.cleanup()
        assert torch.cuda.memory_allocated() == 0 