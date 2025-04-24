# API-Dokumentation

## 📥 Data Module

### `loader.py`

```python
def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lädt historische Marktdaten von Yahoo Finance.
    
    Args:
        symbol: Aktien-Symbol (z.B. "AAPL")
        start_date: Startdatum im Format "YYYY-MM-DD"
        end_date: Enddatum im Format "YYYY-MM-DD"
        
    Returns:
        DataFrame mit OHLCV-Daten
    """
```

## 🧮 Features Module

### `indicators.py`

```python
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt alle technischen Indikatoren zum DataFrame hinzu.
    
    Args:
        df: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit zusätzlichen Indikatoren
    """

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Berechnet den Relative Strength Index (RSI).
    
    Args:
        data: Preiszeitreihe
        period: RSI-Periode (Standard: 14)
        
    Returns:
        RSI-Werte als Series
    """
```

## 🤖 Model Module

### `builder.py`

```python
def build_lstm_model(
    sequence_length: int,
    n_features: int,
    n_units: int = 50
) -> tf.keras.Model:
    """
    Erstellt ein LSTM-Modell für Zeitreihenvorhersage.
    
    Args:
        sequence_length: Länge der Eingabesequenz
        n_features: Anzahl der Features
        n_units: Anzahl der LSTM-Einheiten
        
    Returns:
        Kompiliertes LSTM-Modell
    """
```

### `trainer.py`

```python
def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32
) -> tf.keras.callbacks.History:
    """
    Trainiert das LSTM-Modell.
    
    Args:
        model: LSTM-Modell
        X_train: Trainingsdaten
        y_train: Trainingslabels
        epochs: Anzahl der Epochen
        batch_size: Batch-Größe
        
    Returns:
        Trainingshistorie
    """
```

## 📈 Signals Module

### `logic.py`

```python
def generate_signals(
    predictions: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Generiert Handelssignale aus Modellvorhersagen.
    
    Args:
        predictions: Modellvorhersagen
        threshold: Signalschwelle
        
    Returns:
        Array mit Handelssignalen (-1: Verkauf, 0: Halten, 1: Kauf)
    """
```

## 🎨 Visual Module

### `plotter.py`

```python
def plot_signals(
    df: pd.DataFrame,
    signals: np.ndarray,
    title: str = "Handelssignale"
) -> go.Figure:
    """
    Erstellt interaktiven Plot mit Preisen und Signalen.
    
    Args:
        df: DataFrame mit OHLCV-Daten
        signals: Array mit Handelssignalen
        title: Plot-Titel
        
    Returns:
        Plotly-Figure
    """
```

## 🧪 Backtest Module

### `simulator.py`

```python
def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Führt Backtest der Handelsstrategie durch.
    
    Args:
        df: DataFrame mit OHLCV-Daten
        signals: Array mit Handelssignalen
        initial_capital: Startkapital
        
    Returns:
        Dictionary mit Backtesting-Ergebnissen
    """
``` 