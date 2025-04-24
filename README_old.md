# ML4T - Machine Learning für Trading

Ein modernes Framework für die Entwicklung, das Testen und die Optimierung von Trading-Strategien mit maschinellem Lernen.

## Features

- **LSTM-basierte Vorhersagemodelle**: Implementierung von Deep Learning Modellen für Zeitreihenvorhersagen
- **A/B Testing Framework**: Robuste Vergleiche verschiedener Modellvarianten
- **Signalgenerierung**: Flexible Generierung von Trading-Signalen basierend auf Modellvorhersagen
- **Backtesting**: Umfassende Backtesting-Funktionalität mit detaillierten Performance-Metriken
- **Visualisierung**: Interaktive Visualisierungen mit Plotly

## Installation

1. Repository klonen:
```bash
git clone https://github.com/yourusername/ml4t_project.git
cd ml4t_project
```

2. Python-Umgebung erstellen und Abhängigkeiten installieren:
```bash
python -m venv .venv
source .venv/bin/activate  # Unter Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Projektstruktur

```
ml4t_project/
├── backtest/           # Backtesting-Funktionalität
│   └── simulator.py    # Backtesting-Engine
├── model/             # ML-Modelle
│   ├── ab_testing.py  # A/B Testing Framework
│   └── builder.py     # Modellarchitekturen
├── signals/           # Signalgenerierung
│   └── logic.py      # Trading-Signale
├── visual/           # Visualisierung
│   └── plotter.py    # Plotting-Funktionen
└── tests/            # Testsuite
    └── test_*.py     # Testmodule
```

## Verwendung

### A/B Testing von Modellen

```python
from ml4t_project.model.ab_testing import ABTesting

# Experiment initialisieren
ab_test = ABTesting("lstm_comparison")

# Varianten hinzufügen
ab_test.add_variant(
    name="model_a",
    hyperparameters={'n_units': 64},
    sequence_length=10,
    n_features=2
)

ab_test.add_variant(
    name="model_b",
    hyperparameters={'n_units': 128},
    sequence_length=10,
    n_features=2
)

# Varianten evaluieren
ab_test.evaluate_variant("model_a", test_data)
ab_test.evaluate_variant("model_b", test_data)

# Varianten vergleichen
t_stat, p_value = ab_test.compare_variants("model_a", "model_b")

# Beste Variante ermitteln
best_variant = ab_test.get_best_variant()
```

### Backtesting

```python
from ml4t_project.backtest.simulator import run_backtest
from ml4t_project.signals.logic import generate_signals

# Signale generieren
signals = generate_signals(predictions)

# Backtest durchführen
results = run_backtest(market_data, signals)

print(f"Gesamtrendite: {results['return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
```

## Tests

Die Testsuite kann mit pytest ausgeführt werden:

```bash
pytest tests/ -v
```

## Beitragen

1. Fork erstellen
2. Feature Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add some AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request erstellen

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE) Datei für Details.

## Kontakt

Ihr Name - [@IhrTwitter](https://twitter.com/IhrTwitter) - email@example.com

Projekt Link: [https://github.com/yourusername/ml4t_project](https://github.com/yourusername/ml4t_project) 