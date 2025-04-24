"""
Tests für das Distributed Training Framework
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from ml4t_project.training.distributed_training import DistributedTrainer
import os
from pathlib import Path
import shutil

class SimpleModel(nn.Module):
    """Einfaches Modell für Tests"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

class SimpleDataset(torch.utils.data.Dataset):
    """Einfacher Datensatz für Tests"""
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def dataset():
    return SimpleDataset()

@pytest.fixture
def trainer(model):
    return DistributedTrainer(model)

@pytest.fixture(autouse=True)
def cleanup():
    """Bereinigt Testdateien nach jedem Test"""
    yield
    checkpoint_dir = Path("ml4t_project/exports/checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

def test_trainer_initialization(trainer):
    """Testet die Initialisierung des Trainers"""
    assert trainer.num_gpus == torch.cuda.device_count()
    assert trainer.master_addr == 'localhost'
    assert trainer.master_port == '12355'
    assert isinstance(trainer.metrics, dict)

def test_data_preparation(trainer, dataset):
    """Testet die Datenvorbereitung"""
    world_size = 2
    rank = 0
    batch_size = 32
    
    dataloader = trainer.prepare_data(dataset, batch_size, rank, world_size)
    
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert isinstance(dataloader.sampler, torch.utils.data.DistributedSampler)
    assert dataloader.batch_size == batch_size

@pytest.mark.skipif(torch.cuda.device_count() < 2, 
                   reason="Benötigt mindestens 2 GPUs")
def test_distributed_training(trainer, dataset):
    """Testet das verteilte Training"""
    world_size = 2
    
    # Starte Trainingsprozesse
    import torch.multiprocessing as mp
    mp.spawn(
        trainer.train_model,
        args=(world_size, dataset, None, 2, 32, 0.001),
        nprocs=world_size,
        join=True
    )
    
    # Überprüfe Checkpoints
    checkpoint_dir = Path("ml4t_project/exports/checkpoints")
    assert checkpoint_dir.exists()
    assert any(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    assert (checkpoint_dir / 'training_metrics.json').exists()

def test_checkpoint_saving_loading(trainer, model):
    """Testet das Speichern und Laden von Checkpoints"""
    # Erstelle einen Dummy-Checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    trainer.save_checkpoint(model, optimizer, 0)
    
    # Überprüfe Checkpoint-Dateien
    checkpoint_dir = Path("ml4t_project/exports/checkpoints")
    assert (checkpoint_dir / 'checkpoint_epoch_0.pt').exists()
    assert (checkpoint_dir / 'training_metrics.json').exists()
    
    # Lade Checkpoint
    checkpoint = trainer.load_checkpoint(0)
    assert 'epoch' in checkpoint
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'metrics' in checkpoint

def test_error_handling(trainer):
    """Testet die Fehlerbehandlung"""
    # Test: Laden eines nicht existierenden Checkpoints
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(999)
    
    # Test: Laden ohne vorhandene Checkpoints
    checkpoint_dir = Path("ml4t_project/exports/checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint() 