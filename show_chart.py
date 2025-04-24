import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from ml4t_project.models.lstm_model import LSTMPredictor
import torch

def get_lstm_predictions(df, lookback=60):
    """
    Generiert LSTM-Vorhersagen für den Datensatz
    
    Args:
        df: DataFrame mit den Kursdaten
        lookback: Anzahl der Tage für die Vorhersage
        
    Returns:
        numpy.array: Array mit Vorhersagen
    """
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

def main(symbol="AAPL", start_date="2022-01-01", end_date="2024-01-01"):
    """
    Hauptfunktion für die technische Analyse und Chartgenerierung
    
    Args:
        symbol: Aktien-Symbol (default: AAPL)
        start_date: Startdatum (default: 2022-01-01)
        end_date: Enddatum (default: 2024-01-01)
    """
    # Daten abrufen und Debug-Ausgaben
    print(f"\nLade Daten für {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date)

    # Entferne Multi-Index falls vorhanden
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"Datenpunkte geladen: {len(df)}")
    print("\nDatenqualität:")
    print(f"Fehlende Werte: {df.isnull().sum().sum()}")
    print(f"Preisbereich: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    print(f"Handelstage: {df.index[0].strftime('%Y-%m-%d')} bis {df.index[-1].strftime('%Y-%m-%d')}")

    # Berechne technische Indikatoren
    print("\nBerechne technische Indikatoren...")
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))

    # Generiere Handelssignale
    print("\nGeneriere Handelssignale...")
    df['Signal'] = 0  # 0: Halten, 1: Kaufen, -1: Verkaufen

    # Kaufsignal: MA20 kreuzt MA50 von unten + RSI < 70
    df['Signal'] = np.where((df['MA20'] > df['MA50']) & 
                           (df['MA20'].shift(1) <= df['MA50'].shift(1)) & 
                           (df['RSI'] < 70), 1, df['Signal'])

    # Verkaufssignal: MA20 kreuzt MA50 von oben + RSI > 30
    df['Signal'] = np.where((df['MA20'] < df['MA50']) & 
                           (df['MA20'].shift(1) >= df['MA50'].shift(1)) & 
                           (df['RSI'] > 30), -1, df['Signal'])

    # LSTM-Vorhersagen
    print("\nGeneriere LSTM-Vorhersagen...")
    predictions = get_lstm_predictions(df)
    
    # Füge Vorhersagen zum DataFrame hinzu
    df['LSTM_Prediction'] = np.nan
    df.iloc[60:len(predictions)+60, df.columns.get_loc('LSTM_Prediction')] = predictions.flatten()

    # Erstelle Figure
    print("\nErstelle Chart...")
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

    # Kaufsignale mit Annotation
    buy_signals = df[df['Signal'] == 1]
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

    # Verkaufsignale mit Annotation
    sell_signals = df[df['Signal'] == -1]
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
            text=f'{symbol} Aktienkurs mit Handelssignalen ({start_date} bis {end_date})',
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
    output_file = f'aktien_chart_{symbol}.html'
    fig.write_html(output_file, auto_open=False, include_plotlyjs=True)
    print(f"Chart wurde gespeichert in: {output_file}")

    # Statistiken zu den Handelssignalen
    print("\nHandelssignale Statistiken:")
    print(f"Anzahl Kaufsignale: {len(buy_signals)}")
    print(f"Anzahl Verkaufsignale: {len(sell_signals)}")

    # Berechne Performance der Signale
    print("\nSignal-Performance:")
    for idx, row in buy_signals.iterrows():
        # Suche nächstes Verkaufssignal
        next_sell = sell_signals[sell_signals.index > idx].index[0] if len(sell_signals[sell_signals.index > idx]) > 0 else df.index[-1]
        buy_price = row['Close']
        sell_price = df.loc[next_sell, 'Close']
        profit = (sell_price - buy_price) / buy_price * 100
        print(f"Signal vom {idx.strftime('%Y-%m-%d')}: Kauf @ ${buy_price:.2f}, Verkauf @ ${sell_price:.2f}, Gewinn: {profit:.1f}%")

    return fig

if __name__ == "__main__":
    main() 