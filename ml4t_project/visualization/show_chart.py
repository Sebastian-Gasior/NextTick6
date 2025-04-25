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
from ml4t_project.models.lstm_trainer import LSTMTrainer
import torch
import logging
import argparse
from plotly.subplots import make_subplots

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
            
        # Initialisiere Trainer
        trainer = LSTMTrainer(
            input_dim=1,
            hidden_dim=64,
            num_layers=2,
            output_dim=1,
            batch_size=32,
            epochs=100
        )
        
        # Versuche gespeichertes Modell zu laden
        if not trainer.load_model(df.name if hasattr(df, 'name') else 'UNKNOWN'):
            # Wenn kein Modell gefunden, trainiere ein neues
            logger.info("Kein trainiertes Modell gefunden. Starte Training...")
            trainer.train(df, df.name if hasattr(df, 'name') else 'UNKNOWN', lookback)
        
        # Daten vorbereiten
        data = trainer.scaler.transform(df[['Close']].values)
        
        # Vorhersagen generieren
        predictions = []
        for i in range(lookback, len(data)):
            sequence = data[i-lookback:i]
            sequence = torch.FloatTensor(sequence).unsqueeze(0)
            pred = trainer.predict(sequence)
            predictions.append(pred[0])
        
        return np.array(predictions)
        
    except Exception as e:
        logger.error(f"Fehler bei der LSTM-Vorhersage: {str(e)}")
        raise

def create_chart_for_dashboard(symbol, start_date, end_date, lstm_predictions=None, show_signals=False, is_detailed=False):
    """
    Erstellt einen Chart für die Dashboard-Integration
    
    Args:
        symbol: Aktien-Symbol
        start_date: Startdatum 
        end_date: Enddatum
        lstm_predictions: Optional, vorberechnete LSTM-Vorhersagen
        show_signals: Ob Handelssignale angezeigt werden sollen
        is_detailed: Ob es sich um die detaillierte Ansicht handelt
        
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
        
        # Berechne MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Berechne Handelssignale
        df['Signal'] = 0
        df.loc[(df['Close'] > df['MA20']) & (df['MA20'] > df['MA50']), 'Signal'] = 1  # Kaufsignal
        df.loc[(df['Close'] < df['MA20']) & (df['MA20'] < df['MA50']), 'Signal'] = -1  # Verkaufsignal
        
        # LSTM-Vorhersagen verwenden oder generieren
        if lstm_predictions is not None:
            predictions = lstm_predictions
        else:
            predictions = get_lstm_predictions(df)
            
        # Füge Vorhersagen zum DataFrame hinzu
        df['LSTM_Prediction'] = np.nan
        df.iloc[60:len(predictions)+60, df.columns.get_loc('LSTM_Prediction')] = predictions.flatten()
        
        # Berechne Performance-Metriken
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max()
        
        # Erstelle Figure
        if is_detailed:
            # Erstelle subplot-basierte Detailansicht
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Hauptchart mit Kerzen und Indikatoren
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_middle'],
                name='BB Middle',
                line=dict(color='gray', width=1),
                opacity=0.5
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5,
                fill='tonexty'
            ), row=1, col=1)
            
            # Gleitende Durchschnitte
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA50'],
                name='50-Tage MA',
                line=dict(color='#ff9800', width=1.5)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA20'],
                name='20-Tage MA',
                line=dict(color='#2196f3', width=1.5)
            ), row=1, col=1)
            
            # LSTM-Vorhersagen
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['LSTM_Prediction'],
                name='LSTM-Vorhersage',
                line=dict(color='#e91e63', width=2),
                opacity=0.7
            ), row=1, col=1)
            
            # Volumen
            colors = np.where(df['Close'] >= df['Open'], '#26a69a', '#ef5350')
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volumen',
                marker_color=colors,
                opacity=0.5
            ), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=1)
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Signal'],
                name='Signal',
                line=dict(color='orange', width=1)
            ), row=4, col=1)
            
            # Performance-Metriken als Annotation
            metrics_text = (
                f'Volatilität: {volatility:.2%}<br>'
                f'Sharpe Ratio: {sharpe:.2f}<br>'
                f'Max Drawdown: {max_drawdown:.2%}'
            )
            
            fig.add_annotation(
                text=metrics_text,
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.02,
                y=0.98,
                bordercolor='gray',
                borderwidth=1,
                bgcolor='white'
            )
            
            if show_signals:
                # Kaufsignale
                buy_signals = df[df['Signal'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Low'] * 0.98,
                        name='Kaufsignal',
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up',
                            size=16,
                            color='#26a69a',
                            line=dict(width=2, color='white')
                        ),
                        text=['KAUF' for _ in range(len(buy_signals))],
                        textposition='bottom center',
                        textfont=dict(size=12, color='#26a69a')
                    ), row=1, col=1)
                
                # Verkaufsignale
                sell_signals = df[df['Signal'] == -1]
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['High'] * 1.02,
                        name='Verkaufsignal',
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=16,
                            color='#ef5350',
                            line=dict(width=2, color='white')
                        ),
                        text=['VERKAUF' for _ in range(len(sell_signals))],
                        textposition='top center',
                        textfont=dict(size=12, color='#ef5350')
                    ), row=1, col=1)
            
            # Layout für detaillierte Ansicht
            title_text = f'{symbol} Detaillierte Technische Analyse mit LSTM-Vorhersage'
            
            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    font=dict(size=20)
                ),
                height=1000,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Achsenbeschriftungen
            fig.update_yaxes(title_text="Preis in USD", row=1, col=1)
            fig.update_yaxes(title_text="Volumen", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            fig.update_xaxes(title_text="Datum", row=4, col=1)
            
        else:
            # Einfache Ansicht (bisheriger Code)
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
            
            # LSTM-Vorhersagen
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['LSTM_Prediction'],
                name='LSTM-Vorhersage',
                line=dict(color='#e91e63', width=2),
                opacity=0.7
            ))
            
            if show_signals:
                # Kaufsignale
                buy_signals = df[df['Signal'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Low'] * 0.98,
                        name='Kaufsignal',
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up',
                            size=16,
                            color='#26a69a',
                            line=dict(width=2, color='white')
                        ),
                        text=['KAUF' for _ in range(len(buy_signals))],
                        textposition='bottom center',
                        textfont=dict(size=12, color='#26a69a')
                    ))
                
                # Verkaufsignale
                sell_signals = df[df['Signal'] == -1]
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['High'] * 1.02,
                        name='Verkaufsignal',
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=16,
                            color='#ef5350',
                            line=dict(width=2, color='white')
                        ),
                        text=['VERKAUF' for _ in range(len(sell_signals))],
                        textposition='top center',
                        textfont=dict(size=12, color='#ef5350')
                    ))
            
            # Layout
            min_price = df['Low'].min() * 0.95
            max_price = df['High'].max() * 1.05
            
            title_text = f'{symbol} Aktienkurs mit LSTM-Vorhersage'
            
            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    font=dict(size=18)
                ),
                yaxis=dict(
                    title='Preis in USD',
                    tickformat='$.2f',
                    gridcolor='rgba(255,255,255,0.1)',
                    range=[min_price, max_price]
                ),
                xaxis=dict(
                    title='Datum',
                    gridcolor='rgba(255,255,255,0.1)',
                    rangeslider=dict(visible=False)
                ),
                margin=dict(t=50, b=50),
                height=600 if is_detailed else 400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                template='plotly_white',
                hovermode='x unified'
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