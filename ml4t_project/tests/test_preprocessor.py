"""
Test-Modul für Datenvorverarbeitung
"""
import pytest
import numpy as np
import pandas as pd
from ml4t_project.data.preprocessor import normalize_data, create_sequences, prepare_data

@pytest.fixture
def sample_data():
    """Beispieldaten für Tests"""
    return pd.DataFrame({
        'Open': [100.0, 102.0, 101.0, 103.0, 102.0],
        'High': [103.0, 104.0, 103.0, 105.0, 104.0],
        'Low': [98.0, 100.0, 99.0, 101.0, 100.0],
        'Close': [102.0, 101.0, 103.0, 102.0, 103.0],
        'Volume': [1000, 1100, 900, 1200, 1000]
    })

def test_normalize_data(sample_data):
    """Test der Datennormalisierung"""
    # Setup
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Ausführung
    df_norm, scalers = normalize_data(sample_data, columns)
    
    # Assertions
    assert isinstance(df_norm, pd.DataFrame), "Rückgabe sollte DataFrame sein"
    assert isinstance(scalers, dict), "Scalers sollten als Dictionary zurückgegeben werden"
    assert all(col in df_norm.columns for col in columns), "Alle Spalten sollten erhalten bleiben"
    assert all(0 <= df_norm[col].min() and df_norm[col].max() <= 1 for col in columns), \
        "Werte sollten zwischen 0 und 1 normalisiert sein"

def test_create_sequences():
    """Test der Sequenzerstellung"""
    # Setup
    data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0]
    ])
    window_size = 2
    
    # Ausführung
    X, y = create_sequences(data, window_size)
    
    # Assertions
    assert isinstance(X, np.ndarray), "X sollte ein numpy array sein"
    assert isinstance(y, np.ndarray), "y sollte ein numpy array sein"
    assert X.shape == (3, 2, 2), f"Erwartete Shape (3,2,2), bekam {X.shape}"
    assert y.shape == (3,), f"Erwartete Shape (3,), bekam {y.shape}"
    assert np.array_equal(X[0], np.array([[1.0, 2.0], [2.0, 3.0]])), "Erste Sequenz falsch"
    assert y[0] == 3.0, "Erstes Target falsch"

def test_prepare_data(sample_data):
    """Test der Datenvorbereitung"""
    # Setup
    window_size = 2
    train_split = 0.8
    
    # Ausführung
    X_train, X_test, y_train, y_test = prepare_data(
        sample_data, 
        window_size=window_size,
        train_split=train_split
    )
    
    # Assertions
    assert isinstance(X_train, np.ndarray), "X_train sollte numpy array sein"
    assert isinstance(X_test, np.ndarray), "X_test sollte numpy array sein"
    assert isinstance(y_train, np.ndarray), "y_train sollte numpy array sein"
    assert isinstance(y_test, np.ndarray), "y_test sollte numpy array sein"
    
    # Überprüfe Shapes
    total_sequences = len(sample_data) - window_size
    train_size = int(total_sequences * train_split)
    assert len(X_train) == train_size, "Falsche Trainingsdaten-Größe"
    assert len(X_test) == total_sequences - train_size, "Falsche Testdaten-Größe"

def test_prepare_data_invalid():
    """Test der Fehlerbehandlung bei der Datenvorbereitung"""
    # Setup
    invalid_data = pd.DataFrame({
        'Wrong': [1, 2, 3]
    })
    
    # Assertion
    with pytest.raises(KeyError):
        prepare_data(invalid_data, window_size=2)

def test_normalize_data_empty():
    """Test der Normalisierung mit leeren Daten"""
    # Setup
    empty_df = pd.DataFrame()
    
    # Assertion
    with pytest.raises(ValueError):
        normalize_data(empty_df, ['Open'])

def test_create_sequences_invalid_window():
    """Test der Sequenzerstellung mit ungültigem Fenster"""
    # Setup
    data = np.array([[1.0], [2.0]])
    
    # Assertion
    with pytest.raises(ValueError):
        create_sequences(data, window_size=3)  # Fenster größer als Daten

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 