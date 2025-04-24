"""
Tests für das LSTM-Modell
"""

import pytest
import torch
import numpy as np
from ml4t_project.models.lstm_model import LSTMPredictor

@pytest.fixture
def model_params():
    """Fixture für Modellparameter"""
    return {
        'input_dim': 10,
        'hidden_dim': 32,
        'num_layers': 2,
        'output_dim': 1,
        'dropout': 0.2
    }

@pytest.fixture
def sample_data():
    """Fixture für Testdaten"""
    batch_size = 16
    seq_length = 20
    input_dim = 10
    
    X = torch.randn(batch_size, seq_length, input_dim)
    y = torch.randn(batch_size, 1)
    
    return X, y

def test_model_initialization(model_params):
    """Test der Modellinitialisierung"""
    model = LSTMPredictor(**model_params)
    
    assert isinstance(model, LSTMPredictor)
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert isinstance(model.fc, torch.nn.Linear)
    
    assert model.lstm.input_size == model_params['input_dim']
    assert model.lstm.hidden_size == model_params['hidden_dim']
    assert model.lstm.num_layers == model_params['num_layers']
    
    assert model.fc.in_features == model_params['hidden_dim']
    assert model.fc.out_features == model_params['output_dim']

def test_forward_pass(model_params, sample_data):
    """Test des Forward-Passes"""
    model = LSTMPredictor(**model_params)
    X, _ = sample_data
    
    # Forward Pass
    predictions, hidden = model(X)
    
    # Überprüfe Output-Dimensionen
    assert predictions.shape == (X.shape[0], model_params['output_dim'])
    assert len(hidden) == 2  # (hidden_state, cell_state)
    assert hidden[0].shape == (
        model_params['num_layers'],
        X.shape[0],
        model_params['hidden_dim']
    )

def test_init_hidden(model_params):
    """Test der Hidden State Initialisierung"""
    model = LSTMPredictor(**model_params)
    batch_size = 16
    device = torch.device('cpu')
    
    hidden = model.init_hidden(batch_size, device)
    
    assert len(hidden) == 2
    assert hidden[0].shape == (
        model_params['num_layers'],
        batch_size,
        model_params['hidden_dim']
    )
    assert hidden[1].shape == (
        model_params['num_layers'],
        batch_size,
        model_params['hidden_dim']
    )

def test_predict_sequence(model_params):
    """Test der Sequenzvorhersage"""
    model = LSTMPredictor(**model_params)
    batch_size = 1
    seq_length = 20
    prediction_length = 10
    device = torch.device('cpu')
    
    # Erstelle Testsequenz
    initial_sequence = torch.randn(
        batch_size,
        seq_length,
        model_params['input_dim']
    )
    
    # Generiere Vorhersagen
    predictions = model.predict_sequence(
        initial_sequence,
        prediction_length,
        device
    )
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (prediction_length, model_params['output_dim'])

def test_model_training_mode(model_params, sample_data):
    """Test des Trainings- und Evaluationsmodus"""
    model = LSTMPredictor(**model_params)
    X, _ = sample_data
    
    # Test Training Mode
    model.train()
    assert model.training
    
    # Forward Pass im Training Mode
    _, _ = model(X)
    
    # Test Evaluation Mode
    model.eval()
    assert not model.training
    
    # Forward Pass im Eval Mode
    with torch.no_grad():
        _, _ = model(X)

def test_dropout(model_params, sample_data):
    """Test des Dropout-Verhaltens"""
    # Modell mit und ohne Dropout
    model_with_dropout = LSTMPredictor(**model_params)
    model_params['dropout'] = 0.0
    model_without_dropout = LSTMPredictor(**model_params)
    
    X, _ = sample_data
    
    # Training Mode (Dropout aktiv)
    model_with_dropout.train()
    out1_train, _ = model_with_dropout(X)
    out2_train, _ = model_with_dropout(X)
    
    # Outputs sollten unterschiedlich sein im Training
    assert not torch.allclose(out1_train, out2_train)
    
    # Eval Mode (Dropout inaktiv)
    model_with_dropout.eval()
    with torch.no_grad():
        out1_eval, _ = model_with_dropout(X)
        out2_eval, _ = model_with_dropout(X)
    
    # Outputs sollten gleich sein im Eval Mode
    assert torch.allclose(out1_eval, out2_eval)
    
    # Modell ohne Dropout sollte immer gleiche Outputs haben
    model_without_dropout.train()
    out1_no_dropout, _ = model_without_dropout(X)
    out2_no_dropout, _ = model_without_dropout(X)
    assert torch.allclose(out1_no_dropout, out2_no_dropout)

def test_gpu_support(model_params, sample_data):
    """Test der GPU-Unterstützung (wenn verfügbar)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA nicht verfügbar - überspringe GPU-Tests")
    
    device = torch.device('cuda')
    model = LSTMPredictor(**model_params)
    model.to(device)
    
    X, _ = sample_data
    X = X.to(device)
    
    # Forward Pass auf GPU
    predictions, hidden = model(X)
    
    assert predictions.device.type == 'cuda'
    assert hidden[0].device.type == 'cuda'
    assert hidden[1].device.type == 'cuda' 