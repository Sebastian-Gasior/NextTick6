import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ml4t_project.training.distributed_trainer import DistributedTrainer
from ml4t_project.models.lstm_model import LSTMPredictor
import os

@pytest.fixture
def sample_data():
    """Erstellt Beispieldaten für die Tests."""
    # Erstelle einen kleinen Datensatz mit 10 Sequenzen
    X = torch.randn(10, 5, 3)  # (batch_size, seq_length, features)
    y = torch.randn(10, 1)  # (batch_size, targets)
    return X, y

@pytest.fixture
def model():
    """Erstellt ein LSTM-Modell für die Tests."""
    return LSTMPredictor(
        input_dim=3,
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )

@pytest.mark.skip(reason="Verteiltes Training erfordert spezielle Netzwerkkonfiguration")
def test_distributed_initialization():
    """Testet die Initialisierung des verteilten Trainers."""
    trainer = DistributedTrainer(
        model=LSTMPredictor(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        ),
        world_size=2
    )
    assert trainer.world_size == 2
    assert trainer.rank == 0
    assert trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_forward(sample_data, model):
    """Testet den Forward-Pass des Modells."""
    X, _ = sample_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = X.to(device)
    
    # Forward Pass
    with torch.no_grad():
        outputs, hidden = model(X)
    
    assert outputs.shape == (10, 1)  # (batch_size, output_dim)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2  # (hidden_state, cell_state)
    assert hidden[0].shape == (2, 10, 64)  # (num_layers, batch_size, hidden_dim)
    assert hidden[1].shape == (2, 10, 64)  # (num_layers, batch_size, hidden_dim)

def test_model_training_step(sample_data, model):
    """Testet einen einzelnen Trainingsschritt."""
    X, y = sample_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    # Training Step
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

def test_model_validation(sample_data, model):
    """Testet die Validierung."""
    X, y = sample_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    
    criterion = torch.nn.MSELoss()
    
    # Validation Step
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X)
        loss = criterion(outputs, y)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

def test_model_save_load(model, tmp_path):
    """Testet das Speichern und Laden des Modells."""
    # Speichere das Modell
    save_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), save_path)
    assert save_path.exists()
    
    # Lade das Modell
    loaded_model = LSTMPredictor(
        input_dim=3,
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Vergleiche die Parameter
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)

def test_model_error_handling(sample_data, model):
    """Testet die Fehlerbehandlung."""
    X, y = sample_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Teste mit ungültigen Eingabedimensionen
    with pytest.raises(ValueError, match="Eingabe muss 3-dimensional sein"):
        model(torch.randn(10, 3).to(device))  # Fehlende Sequenzlänge
    
    # Teste mit falscher Feature-Dimension
    with pytest.raises(ValueError, match="Eingabe muss 3 Features haben"):
        model(torch.randn(10, 5, 4).to(device))  # Zu viele Features
    
    # Teste mit leeren Eingaben
    with pytest.raises(RuntimeError, match="Batch darf nicht leer sein"):
        model(torch.randn(0, 5, 3).to(device))  # Leerer Batch 