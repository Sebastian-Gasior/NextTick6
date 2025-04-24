"""
Hauptskript zur Steuerung der ML4T-Pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path

from ml4t_project import config
from ml4t_project.data import loader, preprocessor
from ml4t_project.features import indicators
from ml4t_project.model import builder, trainer, predictor
from ml4t_project.signals import logic
from ml4t_project.visual import plotter
from ml4t_project.backtest import simulator


def main():
    # Verzeichnisse erstellen
    Path("exports").mkdir(exist_ok=True)
    
    # 1. Daten laden
    print("Lade Daten...")
    df = loader.load_data(
        symbol=config.SYMBOL,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    if df is None:
        return
    
    # 2. Feature Engineering
    print("Berechne technische Indikatoren...")
    df = indicators.add_all_indicators(df)
    
    # 3. Preprocessing
    print("Bereite Daten vor...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df,
        window_size=config.WINDOW_SIZE,
        train_split=config.TRAIN_TEST_SPLIT
    )
    
    # 4. Modell erstellen und trainieren
    print("Trainiere Modell...")
    model = builder.create_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=config.LSTM_UNITS,
        dropout=config.DROPOUT_RATE,
        learning_rate=config.LEARNING_RATE
    )
    
    history = trainer.train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE
    )
    
    # 5. Evaluation
    print("Evaluiere Modell...")
    mse, mae = trainer.evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # 6. Vorhersagen & Signale
    print("Generiere Handelssignale...")
    predictions = predictor.predict_sequences(model, X_test, None)  # TODO: Scaler hinzufügen
    signals = logic.generate_signals(
        predictions=predictions,
        prices=y_test,
        buy_threshold=config.BUY_THRESHOLD,
        sell_threshold=config.SELL_THRESHOLD
    )
    
    # 7. Backtest
    print("Führe Backtest durch...")
    results_df, metrics = simulator.run_backtest(
        prices=y_test,
        signals=signals,
        initial_capital=config.INITIAL_CAPITAL,
        position_size=config.POSITION_SIZE
    )
    
    # 8. Visualisierung
    print("Erstelle Plots...")
    fig_predictions = plotter.plot_predictions(
        dates=df.index[-len(y_test):],
        actual_prices=y_test,
        predicted_prices=predictions
    )
    fig_signals = plotter.plot_signals(
        dates=df.index[-len(signals):],
        prices=y_test,
        signals=signals
    )
    fig_performance = plotter.plot_performance(
        dates=df.index[-len(results_df):],
        portfolio_value=results_df["portfolio_value"].values,
        benchmark=y_test
    )
    
    # 9. Export
    print("Speichere Ergebnisse...")
    model.save(config.MODEL_PATH)
    results_df.to_csv(config.RESULTS_PATH)
    fig_performance.write_html(config.PLOT_PATH)
    
    print("Pipeline abgeschlossen!")


if __name__ == "__main__":
    main() 