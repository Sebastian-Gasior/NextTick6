"""
Tests für den Model Validator
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import shutil
import json
import matplotlib.pyplot as plt

from ml4t_project.models.lstm_model import LSTMPredictor
from ml4t_project.models.model_validator import ModelValidator

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
def validator(model):
    """Fixture für den Model Validator"""
    return ModelValidator(model)

@pytest.fixture
def sample_data():
    """Fixture für Testdaten"""
    np.random.seed(42)
    n_samples = 100
    
    # Generiere synthetische Daten
    X = np.random.randn(n_samples, 20, 10)  # (samples, sequence_length, features)
    y = np.random.randn(n_samples, 1)  # (samples, output_dim)
    
    return X, y

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture für temporäres Output-Verzeichnis"""
    output_dir = tmp_path / "test_validation"
    output_dir.mkdir()
    yield output_dir
    # Cleanup
    shutil.rmtree(output_dir)

def test_validator_initialization(validator):
    """Test der Validator-Initialisierung"""
    assert isinstance(validator.model, LSTMPredictor)
    assert validator.device in [torch.device('cuda'), torch.device('cpu')]

def test_calculate_metrics(validator):
    """Test der Metrikberechnung"""
    # Generiere Testdaten
    n_samples = 100
    y_true = np.random.randn(n_samples, 1)
    y_pred = y_true + np.random.normal(0, 0.1, (n_samples, 1))  # Füge Rauschen hinzu
    
    metrics = validator.calculate_metrics(y_true, y_pred)
    
    # Überprüfe Metriken
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'direction_accuracy' in metrics
    
    # Überprüfe Metrikwerte
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert 0 <= metrics['r2'] <= 1
    assert 0 <= metrics['direction_accuracy'] <= 1

def test_plot_predictions(validator, temp_output_dir):
    """Test der Vorhersage-Plots"""
    # Generiere Testdaten
    n_samples = 100
    y_true = np.random.randn(n_samples, 1)
    y_pred = y_true + np.random.normal(0, 0.1, (n_samples, 1))
    
    # Test ohne Speichern
    validator.plot_predictions(y_true, y_pred)
    plt.close()
    
    # Test mit Speichern
    save_path = temp_output_dir / "test_plot.png"
    validator.plot_predictions(y_true, y_pred, save_path=str(save_path))
    
    assert save_path.exists()

def test_validate_model(validator, sample_data, temp_output_dir):
    """Test der Modellvalidierung"""
    X_test, y_test = sample_data
    
    # Führe Validierung durch
    report = validator.validate_model(
        X_test,
        y_test,
        output_dir=str(temp_output_dir)
    )
    
    # Überprüfe Report
    assert isinstance(report, dict)
    assert 'metrics' in report
    assert 'plots' in report
    assert 'timestamp' in report
    assert 'model_info' in report
    
    # Überprüfe gespeicherte Dateien
    metrics_files = list(temp_output_dir.glob("validation_metrics_*.json"))
    plot_files = list(temp_output_dir.glob("prediction_plot_*.png"))
    report_files = list(temp_output_dir.glob("validation_report_*.json"))
    
    assert len(metrics_files) > 0
    assert len(plot_files) > 0
    assert len(report_files) > 0
    
    # Überprüfe JSON-Dateien
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    assert isinstance(metrics, dict)
    
    with open(report_files[0]) as f:
        saved_report = json.load(f)
    assert isinstance(saved_report, dict)

def test_cross_validate(validator, sample_data, temp_output_dir):
    """Test der Kreuzvalidierung"""
    X, y = sample_data
    n_splits = 3
    
    # Führe Kreuzvalidierung durch
    cv_report = validator.cross_validate(
        X,
        y,
        n_splits=n_splits,
        output_dir=str(temp_output_dir)
    )
    
    # Überprüfe Report
    assert isinstance(cv_report, dict)
    assert 'avg_metrics' in cv_report
    assert 'fold_metrics' in cv_report
    assert 'n_splits' in cv_report
    assert 'timestamp' in cv_report
    
    # Überprüfe Folds
    assert len(cv_report['fold_metrics']) == n_splits
    
    # Überprüfe Verzeichnisstruktur
    for fold in range(n_splits):
        fold_dir = temp_output_dir / f"fold_{fold+1}"
        assert fold_dir.exists()
        assert len(list(fold_dir.glob("*.json"))) > 0
        assert len(list(fold_dir.glob("*.png"))) > 0

def test_gpu_validation(validator, sample_data, temp_output_dir):
    """Test der GPU-Validierung (wenn verfügbar)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA nicht verfügbar - überspringe GPU-Tests")
    
    # Verschiebe Modell auf GPU
    validator.model.to('cuda')
    X_test, y_test = sample_data
    
    # Führe Validierung durch
    report = validator.validate_model(
        X_test,
        y_test,
        output_dir=str(temp_output_dir)
    )
    
    assert isinstance(report, dict)
    assert report['model_info']['device'] == 'cuda'

def test_error_handling(validator, sample_data, temp_output_dir):
    """Test der Fehlerbehandlung"""
    X_test, y_test = sample_data
    
    # Test mit ungültigen Daten
    with pytest.raises(Exception):
        validator.validate_model(
            X_test[:, :, :5],  # Falsche Feature-Dimension
            y_test,
            output_dir=str(temp_output_dir)
        )
    
    # Test mit nicht existierendem Verzeichnis
    # Dieser Test verwendet einen willkürlichen Pfad, der:
    # 1. Garantiert nicht existiert (absichtlich ungültiger Pfad)
    # 2. Die Fehlerbehandlung des Validators bei nicht existierenden Verzeichnissen prüft
    # Der Test erwartet eine FileNotFoundError, was das korrekte Verhalten wäre
    non_existent_dir = str(Path("Z:/nicht/existierendes/verzeichnis").resolve())
    with pytest.raises(FileNotFoundError):
        validator.validate_model(
            X_test,
            y_test,
            output_dir=non_existent_dir
        ) 