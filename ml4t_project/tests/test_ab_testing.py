"""
Tests für das A/B Testing Framework
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import torch
from ml4t_project.testing.ab_tester import ABTester

from ml4t_project.model.ab_testing import ABTesting, ModelVariant

def generate_test_data(days: int = 100) -> pd.DataFrame:
    """Generiert Testdaten für die Tests"""
    dates = pd.date_range(start='2020-01-01', periods=days)
    np.random.seed(42)
    
    # Erstelle Testdaten mit OHLCV-Format
    data = pd.DataFrame({
        'Open': np.random.normal(100, 10, days),
        'High': np.random.normal(102, 10, days),
        'Low': np.random.normal(98, 10, days),
        'Close': np.random.normal(100, 10, days),
        'Volume': np.random.normal(1000000, 100000, days)
    }, index=dates)
    
    # Stelle sicher, dass High > Low
    data['High'] = np.maximum(data['High'], data['Low'] + 1)
    
    return data

def prepare_sequences(data: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
    """Bereitet Sequenzen für das LSTM-Modell vor"""
    # Berechne technische Indikatoren als Features
    features = pd.DataFrame(index=data.index)
    features['returns'] = data['Close'].pct_change()
    features['volume_change'] = data['Volume'].pct_change()
    
    # Entferne NaN-Werte
    features = features.dropna()
    
    # Erstelle Sequenzen
    X = []
    for i in range(len(features) - sequence_length):
        X.append(features.iloc[i:i+sequence_length].values)
    
    return np.array(X)

@pytest.fixture
def ab_testing():
    """Fixture für eine ABTesting-Instanz"""
    return ABTesting("test_experiment")

@pytest.fixture
def test_data():
    """Fixture für Testdaten"""
    raw_data = generate_test_data()
    return prepare_sequences(raw_data)

def test_add_variant(ab_testing):
    """Test für das Hinzufügen einer neuen Variante"""
    hyperparameters = {
        'n_units': 64,
        'dropout_rate': 0.3
    }
    
    variant = ab_testing.add_variant(
        name="test_variant",
        hyperparameters=hyperparameters,
        sequence_length=10,
        n_features=2
    )
    
    assert variant.name == "test_variant"
    assert variant.hyperparameters == hyperparameters
    assert variant.metrics == {}
    assert isinstance(variant.creation_date, datetime)
    assert "test_variant" in ab_testing.variants
    assert ab_testing.results["test_variant"] == []

def test_evaluate_variant(ab_testing, test_data):
    """Test für die Evaluierung einer Variante"""
    variant = ab_testing.add_variant(
        name="test_variant",
        hyperparameters={'n_units': 32},
        sequence_length=10,
        n_features=2
    )
    
    metrics = ab_testing.evaluate_variant(
        variant_name="test_variant",
        test_data=test_data
    )
    
    assert isinstance(metrics, dict)
    assert 'mean_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert len(ab_testing.results["test_variant"]) == 1

def test_compare_variants(ab_testing, test_data):
    """Test für den Vergleich von Varianten"""
    # Erste Variante hinzufügen und evaluieren
    ab_testing.add_variant(
        name="variant_a",
        hyperparameters={'n_units': 32},
        sequence_length=10,
        n_features=2
    )
    ab_testing.evaluate_variant("variant_a", test_data)
    
    # Zweite Variante hinzufügen und evaluieren
    ab_testing.add_variant(
        name="variant_b",
        hyperparameters={'n_units': 64},
        sequence_length=10,
        n_features=2
    )
    ab_testing.evaluate_variant("variant_b", test_data)
    
    t_stat, p_value = ab_testing.compare_variants("variant_a", "variant_b")
    
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)
    assert -100 <= t_stat <= 100  # Realistische Grenzen
    assert 0 <= p_value <= 1  # p-Wert muss zwischen 0 und 1 liegen

def test_get_best_variant(ab_testing, test_data):
    """Test für die Ermittlung der besten Variante"""
    # Erste Variante mit schlechteren Ergebnissen
    ab_testing.add_variant(
        name="variant_a",
        hyperparameters={'n_units': 32},
        sequence_length=10,
        n_features=2
    )
    ab_testing.variants["variant_a"].metrics = {'mean_return': 0.05}
    
    # Zweite Variante mit besseren Ergebnissen
    ab_testing.add_variant(
        name="variant_b",
        hyperparameters={'n_units': 64},
        sequence_length=10,
        n_features=2
    )
    ab_testing.variants["variant_b"].metrics = {'mean_return': 0.08}
    
    best_variant = ab_testing.get_best_variant()
    assert best_variant == "variant_b"

def test_save_and_load_results(ab_testing, test_data):
    """Test für das Speichern und Laden von Ergebnissen"""
    # Variante hinzufügen und evaluieren
    ab_testing.add_variant(
        name="test_variant",
        hyperparameters={'n_units': 32},
        sequence_length=10,
        n_features=2
    )
    ab_testing.evaluate_variant("test_variant", test_data)
    
    # Temporäre Datei für den Test erstellen
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filepath = tmp.name
    
    try:
        # Ergebnisse speichern
        ab_testing.save_results(filepath)
        
        # Neue ABTesting-Instanz erstellen und Ergebnisse laden
        new_ab_testing = ABTesting("test_experiment")
        new_ab_testing.load_results(filepath)
        
        # Überprüfen, ob die Daten korrekt geladen wurden
        assert new_ab_testing.experiment_name == ab_testing.experiment_name
        assert list(new_ab_testing.variants.keys()) == list(ab_testing.variants.keys())
        assert list(new_ab_testing.results.keys()) == list(ab_testing.results.keys())
        
    finally:
        # Aufräumen
        os.unlink(filepath)

def test_model_variant_serialization():
    """Test für die Serialisierung von ModelVariant"""
    variant = ModelVariant(
        name="test_variant",
        model=None,
        hyperparameters={'n_units': 32},
        metrics={'mean_return': 0.1},
        creation_date=datetime.now()
    )
    
    # In Dictionary konvertieren
    variant_dict = variant.to_dict()
    
    # Aus Dictionary wiederherstellen
    restored_variant = ModelVariant.from_dict(variant_dict)
    
    assert restored_variant.name == variant.name
    assert restored_variant.hyperparameters == variant.hyperparameters
    assert restored_variant.metrics == variant.metrics
    assert isinstance(restored_variant.creation_date, datetime)

class TestABTester:
    def setup_method(self):
        self.tester = ABTester(
            control_group_size=1000,
            test_group_size=1000,
            duration=timedelta(days=7)
        )
        
    def test_group_assignment(self):
        # Teste Gruppenzuweisung
        user_id = "user123"
        group = self.tester.assign_group(user_id)
        assert group in ['control', 'test']
        
    def test_metric_tracking(self):
        # Teste Metriken-Tracking
        metrics = {
            'conversion_rate': 0.15,
            'revenue': 100.0,
            'engagement': 0.8
        }
        self.tester.track_metrics('test', metrics)
        assert self.tester.get_group_metrics('test')['conversion_rate'] == 0.15
        
    def test_statistical_significance(self):
        # Teste statistische Signifikanz
        control_metrics = {'conversion_rate': 0.1}
        test_metrics = {'conversion_rate': 0.15}
        significance = self.tester.check_significance(control_metrics, test_metrics)
        assert 'p_value' in significance
        assert 'is_significant' in significance
        
    def test_error_handling(self):
        # Teste Fehlerbehandlung
        with pytest.raises(ValueError):
            self.tester.assign_group(None)
            
        with pytest.raises(ValueError):
            self.tester.track_metrics('invalid_group', {})
            
    def test_report_generation(self):
        # Teste Berichtgenerierung
        report = self.tester.generate_report()
        assert 'summary' in report
        assert 'metrics' in report
        assert 'significance' in report
        assert 'recommendations' in report
        
    def test_cleanup(self):
        # Teste Bereinigung
        self.tester.cleanup()
        assert not self.tester.is_active 