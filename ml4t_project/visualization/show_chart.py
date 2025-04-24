"""
Visualisierungskomponente für Aktiencharts und LSTM-Vorhersagen
"""
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from ml4t_project.models.lstm_model import LSTMPredictor
import torch
import logging
import argparse

# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_lstm_predictions(df, lookback=60):
    """
    Generiert LSTM-Vorhersagen für den Datensatz
    
    Args:
        df: DataFrame mit den Kursdaten
        lookback: Anzahl der Tage für die Vorhersage
        
    Returns:
        numpy.array: Array mit Vorhersagen
        
    Raises:
        ValueError: Wenn die Eingabedaten ungültig sind
    """
    try:
        if df.empty:
            raise ValueError("Eingabedaten dürfen nicht leer sein")
            
        if len(df) < lookback:
            raise ValueError(f"Mindestens {lookback} Datenpunkte erforderlich")
            
        # Daten vorbereiten
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']].values)
        
        # LSTM-Modell laden oder initialisieren
        model = LSTMPredictor(
            input_dim=1,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Vorhersagen generieren
        predictions = []
        for i in range(lookback, len(scaled_data)):
            sequence = scaled_data[i-lookback:i]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            with torch.no_grad():
                pred, _ = model(sequence)
            predictions.append(pred.cpu().numpy()[0])
        
        # Skalierung rückgängig machen
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions
        
    except Exception as e:
        logger.error(f"Fehler bei der LSTM-Vorhersage: {str(e)}")
        raise

def create_chart_for_dashboard(symbol, start_date, end_date, lstm_predictions=None):
    """
    Erstellt einen Chart für die Dashboard-Integration
    
    Args:
        symbol: Aktien-Symbol
        start_date: Startdatum 
        end_date: Enddatum
        lstm_predictions: Optional, vorberechnete LSTM-Vorhersagen
        
    Returns:
        Plotly Figure-Objekt
    """
    try:
        # Daten laden
        logger.info(f"Lade Daten für Dashboard-Chart: {symbol}")
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} gefunden")
            
        # Entferne Multi-Index falls vorhanden
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Berechne technische Indikatoren
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                            df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
        
        # LSTM-Vorhersagen verwenden oder generieren
        if lstm_predictions is not None:
            predictions = lstm_predictions
        else:
            predictions = get_lstm_predictions(df)
            
        # Füge Vorhersagen zum DataFrame hinzu
        df['LSTM_Prediction'] = np.nan
        df.iloc[60:len(predictions)+60, df.columns.get_loc('LSTM_Prediction')] = predictions.flatten()
        
        # Erstelle Figure
        fig = go.Figure()
        
        # Candlestick-Chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Gleitende Durchschnitte
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            name='50-Tage MA',
            line=dict(color='#ff9800', width=1.5, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='20-Tage MA',
            line=dict(color='#2196f3', width=1.5, dash='dot')
        ))
        
        # LSTM-Vorhersagen zum Chart hinzufügen
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['LSTM_Prediction'],
            name='LSTM-Vorhersage',
            line=dict(color='#e91e63', width=2),
            opacity=0.7
        ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} Aktienkurs mit LSTM-Vorhersage',
                x=0.5,
                font=dict(size=18)
            ),
            yaxis=dict(
                title='Preis in USD',
                tickformat='$.2f',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            xaxis=dict(
                title='Datum',
                gridcolor='rgba(255,255,255,0.1)',
                rangeslider=dict(visible=False)
            ),
            margin=dict(t=50, b=50),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Dashboard-Chart-Erstellung: {str(e)}")
        raise

def save_chart_to_file(symbol, start_date, end_date, output_dir="ml4t_project/exports/charts"):
    """
    Erstellt und speichert einen Chart für das Dashboard
    
    Args:
        symbol: Aktien-Symbol
        start_date: Startdatum
        end_date: Enddatum
        output_dir: Ausgabeverzeichnis
    
    Returns:
        Pfad zur gespeicherten Datei
    """
    try:
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)
        
        # Erstelle Chart
        fig = main(symbol, start_date, end_date)
        
        # Speichere Chart
        output_file = f"{output_dir}/chart_{symbol}.html"
        fig.write_html(output_file, auto_open=False, include_plotlyjs=True)
        
        logger.info(f"Chart für {symbol} gespeichert in {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Charts: {str(e)}")
        raise

def main(symbol="AAPL", start_date="2022-01-01", end_date="2024-01-01"):
    """
    Hauptfunktion für die technische Analyse und Chartgenerierung
    
    Args:
        symbol: Aktien-Symbol (default: AAPL)
        start_date: Startdatum (default: 2022-01-01)
        end_date: Enddatum (default: 2024-01-01)
        
    Returns:
        plotly.graph_objects.Figure: Generierter Chart
        
    Raises:
        Exception: Bei Fehlern während der Verarbeitung
    """
    try:
        # Daten abrufen und Debug-Ausgaben
        print(f"\nLade Daten für {symbol}...")
        logger.info(f"Lade Daten für {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} gefunden")

        # Entferne Multi-Index falls vorhanden
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print(f"Datenpunkte geladen: {len(df)}")
        print(f"Preisbereich: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        logger.info(f"Datenpunkte geladen: {len(df)}")
        logger.info(f"Preisbereich: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")

        # Berechne technische Indikatoren
        print("\nBerechne technische Indikatoren...")
        logger.info("Berechne technische Indikatoren...")
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))

        # LSTM-Vorhersagen
        print("\nGeneriere LSTM-Vorhersagen...")
        logger.info("Generiere LSTM-Vorhersagen...")
        predictions = get_lstm_predictions(df)
        
        # Füge Vorhersagen zum DataFrame hinzu
        df['LSTM_Prediction'] = np.nan
        df.iloc[60:len(predictions)+60, df.columns.get_loc('LSTM_Prediction')] = predictions.flatten()

        # Erstelle Figure
        print("\nErstelle Chart...")
        logger.info("Erstelle Chart...")
        fig = go.Figure()

        # Volumen
        volume_colors = np.where(df['Close'] >= df['Open'], 'rgba(38,166,154,0.3)', 'rgba(239,83,80,0.3)')
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volumen',
            marker=dict(color=volume_colors),
            yaxis='y2'
        ))

        # Candlestick-Chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))

        # Gleitende Durchschnitte
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            name='50-Tage MA',
            line=dict(color='#ff9800', width=1.5, dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='20-Tage MA',
            line=dict(color='#2196f3', width=1.5, dash='dot')
        ))

        # LSTM-Vorhersagen zum Chart hinzufügen
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['LSTM_Prediction'],
            name='LSTM-Vorhersage',
            line=dict(color='#e91e63', width=2),
            opacity=0.7
        ))

        # Layout
        min_price = df['Low'].min() * 0.95
        max_price = df['High'].max() * 1.05

        fig.update_layout(
            title=dict(
                text=f'{symbol} Aktienkurs mit LSTM-Vorhersage ({start_date} bis {end_date})',
                x=0.5,
                font=dict(size=24)
            ),
            yaxis=dict(
                title='Preis in USD',
                type='linear',
                range=[min_price, max_price],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='$.2f',
                tickfont=dict(size=12),
                side='left',
                autorange=True
            ),
            xaxis=dict(
                title='Datum',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                rangeslider=dict(visible=True),
                tickfont=dict(size=12),
                autorange=True
            ),
            plot_bgcolor='rgb(25,25,25)',
            paper_bgcolor='rgb(25,25,25)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(25,25,25,0.8)'
            ),
            margin=dict(t=100, b=50)
        )

        # Volumen-Achse
        max_vol = df['Volume'].max()
        fig.update_layout(
            yaxis2=dict(
                title=dict(text='Volumen', font=dict(color='#b39ddb')),
                tickfont=dict(color='#b39ddb', size=12),
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, max_vol * 3],
                tickformat=',d',
                autorange=True
            )
        )

        # Speichern
        print("\nSpeichere Chart...")
        logger.info("Speichere Chart...")
        output_file = f'aktien_chart_{symbol}.html'
        fig.write_html(output_file, auto_open=False, include_plotlyjs=True)
        print(f"Chart wurde gespeichert in: {output_file}")
        logger.info(f"Chart wurde gespeichert in: {output_file}")

        return fig

    except Exception as e:
        logger.error(f"Fehler bei der Chartgenerierung: {str(e)}")
        print(f"Fehler bei der Chartgenerierung: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Aktienchart mit LSTM-Vorhersage generieren')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Aktien-Symbol (default: AAPL)')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Startdatum (default: 2022-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='Enddatum (default: 2024-01-01)')
    args = parser.parse_args()
    
    print(f"Generiere Chart für {args.symbol} von {args.start_date} bis {args.end_date}")
    
    # Run the main function
    main(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date) 