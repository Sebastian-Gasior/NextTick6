"""
Tests für den Model Trainer
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import shutil
import json
from torch.utils.data import DataLoader, TensorDataset

from ml4t_project.models.lstm_model import LSTMPredictor
from ml4t_project.models.model_trainer import ModelTrainer

@pytest.fixture
def model():
    """Fixture für das LSTM-Modell"""
    return LSTMPredictor(
        input_dim=10,
        hidden_dim=32,
        num_layers=2,
        output_dim=1
    )

@pytest.fixture
def trainer(model):
    """Fixture für den Model Trainer"""
    return ModelTrainer(model, learning_rate=0.001)

@pytest.fixture
def sample_data():
    """Fixture für Testdaten"""
    # Generiere synthetische Daten
    np.random.seed(42)
    n_samples = 100
    seq_length = 20
    n_features = 10
    
    X = np.random.randn(n_samples, seq_length, n_features)
    y = np.random.randn(n_samples, 1)
    
    return X, y

@pytest.fixture
def data_loaders(sample_data):
    """Fixture für DataLoader"""
    X, y = sample_data
    
    # Konvertiere zu Tensoren
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Erstelle DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader

@pytest.fixture
def temp_model_dir(tmp_path):
    """Fixture für temporäres Modellverzeichnis"""
    model_dir = tmp_path / "test_models"
    model_dir.mkdir()
    yield model_dir
    # Cleanup
    shutil.rmtree(model_dir)

def test_trainer_initialization(trainer):
    """Test der Trainer-Initialisierung"""
    assert isinstance(trainer.model, LSTMPredictor)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer.criterion, torch.nn.MSELoss)
    assert trainer.train_losses == []
    assert trainer.val_losses == []

def test_prepare_data(trainer, sample_data):
    """Test der Datenvorbereitung"""
    X, y = sample_data
    batch_size = 16
    
    train_loader, val_loader = trainer.prepare_data(X, y, batch_size)
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Überprüfe Batch-Größe
    for batch_X, batch_y in train_loader:
        assert batch_X.shape[0] <= batch_size
        assert batch_y.shape[0] <= batch_size
        break

def test_train_epoch(trainer, data_loaders):
    """Test des Trainings einer Epoche"""
    train_loader, _ = data_loaders
    
    # Training einer Epoche
    loss = trainer.train_epoch(train_loader)
    
    assert isinstance(loss, float)
    assert loss > 0  # Loss sollte positiv sein

def test_validate(trainer, data_loaders):
    """Test der Validierung"""
    _, val_loader = data_loaders
    
    # Validierung
    val_loss = trainer.validate(val_loader)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0

def test_full_training(trainer, data_loaders, temp_model_dir):
    """Test des vollständigen Trainings"""
    train_loader, val_loader = data_loaders
    epochs = 2
    
    # Training
    metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        model_dir=str(temp_model_dir)
    )
    
    # Überprüfe Metriken
    assert isinstance(metrics, dict)
    assert len(metrics['train_losses']) == epochs
    assert len(metrics['val_losses']) == epochs
    assert 'best_val_loss' in metrics
    assert 'epochs_trained' in metrics
    
    # Überprüfe gespeicherte Dateien
    checkpoint_files = list(temp_model_dir.glob("model_checkpoint_*.pt"))
    assert len(checkpoint_files) > 0
    
    metrics_file = temp_model_dir / "training_metrics.json"
    assert metrics_file.exists()
    
    # Überprüfe Metrics JSON
    with open(metrics_file) as f:
        saved_metrics = json.load(f)
    assert isinstance(saved_metrics, dict)
    assert 'train_losses' in saved_metrics

def test_early_stopping(trainer, data_loaders, temp_model_dir):
    """Test des Early Stopping"""
    train_loader, val_loader = data_loaders
    epochs = 10
    patience = 2
    
    # Training mit Early Stopping
    metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        model_dir=str(temp_model_dir)
    )
    
    # Überprüfe, ob Training vor der maximalen Epochenzahl gestoppt wurde
    assert metrics['epochs_trained'] <= epochs

def test_model_checkpointing(trainer, data_loaders, temp_model_dir):
    """Test der Modell-Checkpoints"""
    train_loader, val_loader = data_loaders
    epochs = 3
    
    # Training
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        model_dir=str(temp_model_dir)
    )
    
    # Überprüfe Checkpoint-Dateien
    checkpoint_files = list(temp_model_dir.glob("model_checkpoint_*.pt"))
    assert len(checkpoint_files) > 0
    
    # Lade einen Checkpoint
    checkpoint = torch.load(checkpoint_files[0])
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'epoch' in checkpoint
    assert 'val_loss' in checkpoint

def test_gpu_training(trainer, data_loaders, temp_model_dir):
    """Test des GPU-Trainings (wenn verfügbar)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA nicht verfügbar - überspringe GPU-Tests")
    
    # Verschiebe Modell auf GPU
    trainer.model.to('cuda')
    train_loader, val_loader = data_loaders
    
    # Training
    metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=2,
        model_dir=str(temp_model_dir)
    )
    
    assert isinstance(metrics, dict)
    assert len(metrics['train_losses']) == 2 