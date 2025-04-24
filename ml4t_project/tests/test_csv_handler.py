"""
Tests für den CSV-Handler.
"""

import pytest
import pandas as pd
from pathlib import Path
from ml4t_project.data.csv_handler import CSVHandler
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Erstellt ein temporäres Verzeichnis für Tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def csv_handler(temp_dir):
    """Erstellt einen CSV-Handler für Tests."""
    return CSVHandler(temp_dir)

def test_read_write_csv(csv_handler, temp_dir):
    """Testet das Lesen und Schreiben von CSV-Dateien."""
    # Testdaten erstellen
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    # Datei schreiben
    file_path = 'test.csv'
    assert csv_handler.write_csv(data, file_path)
    
    # Datei lesen
    read_data = csv_handler.read_csv(file_path)
    assert read_data is not None
    pd.testing.assert_frame_equal(data, read_data)

def test_list_files(csv_handler, temp_dir):
    """Testet das Auflisten von CSV-Dateien."""
    # Testdateien erstellen
    data = pd.DataFrame({'A': [1]})
    csv_handler.write_csv(data, 'test1.csv')
    csv_handler.write_csv(data, 'subdir/test2.csv')
    
    # Dateien auflisten
    files = csv_handler.list_files()
    assert len(files) == 1
    assert files[0].name == 'test1.csv'
    
    # Rekursiv auflisten
    files = csv_handler.list_files(recursive=True)
    assert len(files) == 2
    assert any(f.name == 'test2.csv' for f in files)

def test_invalid_file_path(csv_handler):
    """Testet das Verhalten bei ungültigen Dateipfaden."""
    # Nicht existierende Datei lesen
    result = csv_handler.read_csv('nonexistent.csv')
    assert result is None
    
    # Ungültige Daten schreiben
    with pytest.raises(ValueError, match="Daten müssen ein pandas DataFrame sein"):
        csv_handler.write_csv(None, 'test.csv')

def test_invalid_data(csv_handler):
    """Testet das Verhalten bei ungültigen Daten."""
    # Leerer DataFrame
    result = csv_handler.write_csv(pd.DataFrame(), 'empty.csv')
    assert result
    
    # Ungültige Datenstruktur
    with pytest.raises(ValueError, match="Daten müssen ein pandas DataFrame sein"):
        csv_handler.write_csv("invalid", 'invalid.csv') 