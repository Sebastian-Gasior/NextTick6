"""
Echtzeit-Trading-Dashboard für ML4T.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import yfinance as yf
import json
import os
from datetime import datetime, timezone
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ml4t_project.analysis.market_analyzer import MarketAnalyzer
from ml4t_project.visualization.show_chart import create_chart_for_dashboard, save_chart_to_file

class MonitoringDashboard(dash.Dash):
    """Trading-Dashboard mit Aktienanalyse und Handelssignalen"""
    
    def __init__(self):
        """Initialisiert das Dashboard mit Standardwerten"""
        self.logger = logging.getLogger(__name__)
        
        # Setze Zeitraum (2 Jahre bis heute) mit UTC
        self.end_date = pd.Timestamp.now(tz='UTC').normalize()  # Normalisiere auf Mitternacht
        self.start_date = (self.end_date - pd.DateOffset(years=2)).normalize()  # Normalisiere auf Mitternacht
        
        self.logger.info(f"Initialisiere Dashboard mit Zeitraum: {self.start_date} bis {self.end_date}")
        
        # Initialisiere Symbole und Layout
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        self.current_symbol = self.symbols[0]
        
        # Erstelle das Dashboard
        super().__init__(__name__)
        
        # Lade initiale Daten
        self._load_data(self.current_symbol)
        
        self.setup_layout()
        self._setup_callbacks()
        
        self.logger.info("Dashboard initialisiert")
        
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Analyzer
        self.market_analyzer = MarketAnalyzer()
        
        # Logging Setup
        logging.basicConfig(
            filename=self.log_dir / "trading.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_layout(self):
        # Layout definieren
        self.layout = html.Div([
            # Header
            html.Div([
                html.H1("ML4T Trading Dashboard", 
                       style={'textAlign': 'center', 'margin': '20px'}),
                
                # Aktienauswahl
                html.Div([
                    html.Label("Aktie auswählen:"),
                    dcc.Dropdown(
                        id='stock-selector',
                        options=[
                            {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                            {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                            {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                            {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
                            {'label': 'Tesla (TSLA)', 'value': 'TSLA'}
                        ],
                        value='AAPL',
                        style={'width': '300px'}
                    )
                ], style={'margin': '20px'})
            ]),
            
            # Hauptbereich - Charts
            html.Div([
                # Kursverlauf
                    html.Div([
                        html.H3("Kursverlauf und Signale"),
                        dcc.Graph(id='price-chart')
                    ], style={'marginBottom': '20px'}),
                    
                    # Technische Indikatoren
                    html.Div([
                        html.H3("Technische Indikatoren"),
                        dcc.Graph(id='indicators-chart')
                ], style={'marginBottom': '20px'})
            ], style={'width': '100%', 'marginBottom': '30px'}),
            
            # Detaillierte LSTM-Vorhersage
            html.Div([
                html.H3("Detaillierte LSTM-Vorhersage"),
                dcc.Graph(
                    id='detailed-lstm-chart',
                    style={'height': '800px'}
                )
            ], style={'width': '100%', 'marginBottom': '30px'}),
            
            # Analysebereich
            html.Div([
                # Technische Analyse
                html.Div([
                    html.H3("Technische Marktanalyse"),
                    html.Pre(id='market-analysis', 
                           style={'padding': '15px', 
                                 'border': '1px solid #ddd', 
                                 'whiteSpace': 'pre-wrap',
                                 'fontFamily': 'monospace', 
                                 'fontSize': '13px', 
                                 'backgroundColor': '#f8f9fa',
                                 'height': '800px',
                                 'overflowY': 'auto',
                                 'margin': '0'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # LSTM-Analyse
                html.Div([
                    html.H3("LSTM-Modell Analyse"),
                    html.Pre(id='lstm-analysis', 
                           style={'padding': '15px', 
                                 'border': '1px solid #ddd', 
                                 'whiteSpace': 'pre-wrap',
                                 'fontFamily': 'monospace', 
                                 'fontSize': '13px', 
                                 'backgroundColor': '#f8f9fa',
                                 'height': '800px',
                                 'overflowY': 'auto',
                                 'margin': '0'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'})
            ], style={'margin': '20px', 'marginBottom': '30px', 'display': 'flex', 'alignItems': 'stretch'}),
            
            # Metriken und Signale
                    html.Div([
                    # Performance-Metriken
                    html.Div([
                        html.H3("Performance-Metriken"),
                        html.Div(id='performance-metrics',
                               style={'padding': '10px', 'border': '1px solid #ddd'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # Handelssignale
                    html.Div([
                        html.H3("Aktuelle Handelssignale"),
                        html.Div(id='trading-signals',
                               style={'padding': '10px', 'border': '1px solid #ddd'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'})
            ], style={'margin': '20px'})
        ], style={'maxWidth': '1800px', 'margin': 'auto', 'padding': '20px'})
    
    def _setup_callbacks(self):
        @self.callback(
            [Output('price-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('detailed-lstm-chart', 'figure'),
             Output('market-analysis', 'children'),
             Output('lstm-analysis', 'children'),
             Output('performance-metrics', 'children'),
             Output('trading-signals', 'children')],
            [Input('stock-selector', 'value')]
        )
        def update_charts(symbol):
            try:
                self.logger.info(f"Aktualisiere Charts für {symbol}")
                
                # Lade Daten mit einheitlichem Zeitraum
                df = self._load_data(symbol)
                if df.empty:
                    raise ValueError(f"Keine Daten für {symbol} verfügbar")
                
                # Debug-Log für DataFrame
                self.logger.info(f"DataFrame Spalten: {df.columns.tolist()}")
                self.logger.info(f"DataFrame Index: {df.index.dtype}")
                self.logger.info(f"Erste Zeilen des DataFrames:\n{df.head()}")
                
                # Berechne technische Indikatoren
                df = self._calculate_technical_indicators(df)
                
                # Berechne Handelssignale
                df = self._calculate_signals(df)
                
                # Erstelle Charts mit einheitlichem Zeitraum
                price_fig = self._create_price_chart(df, symbol)
                indicators_fig = self._create_indicators_chart(df, symbol)
                
                # LSTM-Chart mit gleichem Zeitraum
                detailed_lstm_fig = create_chart_for_dashboard(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    show_signals=True,
                    is_detailed=True
                )
                
                # Generiere LSTM-Vorhersagen
                lstm_predictions = self._generate_lstm_predictions(df)
                df['LSTM_Prediction'] = np.nan
                df.iloc[60:len(lstm_predictions)+60, df.columns.get_loc('LSTM_Prediction')] = lstm_predictions.flatten()
                
                # Generiere Analysen mit gleichem Zeitraum
                market_analysis = self._generate_analysis(df)
                lstm_analysis = self._generate_lstm_analysis(df, symbol)
                
                # Lade Metriken und Signale
                metrics = self._load_metrics(symbol)
                signals = self._load_signals(symbol)
                
                return price_fig, indicators_fig, detailed_lstm_fig, market_analysis, lstm_analysis, metrics, signals
                
            except Exception as e:
                self.logger.error(f"Fehler beim Update der Charts: \n{str(e)}")
                return self._create_error_chart(), self._create_error_chart(), \
                       self._create_error_chart(), \
                       f"Fehler bei der Marktanalyse für {symbol}: {str(e)}", \
                       f"Fehler bei der LSTM-Analyse für {symbol}: {str(e)}", \
                       "Fehler beim Laden der Metriken", \
                       "Fehler beim Laden der Signale"
        @self.callback(
            Output('lstm-iframe', 'src'),
            [Input('stock-selector', 'value')]
        )
        def update_iframe(symbol):
            # Prüfe, ob Chart existiert, sonst generiere ihn
            chart_path = f"ml4t_project/exports/charts/chart_{symbol}.html"
            if not os.path.exists(chart_path):
                try:
                    save_chart_to_file(
                        symbol=symbol,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                except Exception as e:
                    self.logger.error(f"Fehler beim Erstellen des Charts: {str(e)}")
                    return ""
            
            # Relativen Pfad für iframe zurückgeben
            return f"/charts/chart_{symbol}.html"
    
    def _load_data(self, symbol: str) -> pd.DataFrame:
        """
        Lädt Aktiendaten für das angegebene Symbol.
        
        Args:
            symbol: Das Aktiensymbol (z.B. 'AAPL')
            
        Returns:
            pd.DataFrame: DataFrame mit den Aktiendaten oder leerer DataFrame bei Fehler
        """
        try:
            # Stelle sicher, dass Start- und Enddatum in UTC sind
            start_date = self.start_date.tz_localize('UTC') if self.start_date.tz is None else self.start_date
            end_date = self.end_date.tz_localize('UTC') if self.end_date.tz is None else self.end_date
            
            self.logger.info(f"Lade Daten für {symbol} von {start_date} bis {end_date}")
            
            # Lade Daten von yfinance mit expliziten Zeiträumen
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if df.empty:
                self.logger.warning(f"Keine Daten gefunden für {symbol}")
                return pd.DataFrame()
            
            # Debug-Log vor der Verarbeitung
            self.logger.info(f"Rohdaten geladen:\nSpalten: {df.columns}\nIndex: {df.index.dtype}")
            
            # Entferne MultiIndex falls vorhanden
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Stelle sicher, dass der Index im UTC-Format ist
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Filtere nochmal explizit nach dem Zeitraum
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Sortiere nach Datum
            df = df.sort_index()
            
            # Debug-Log nach der Verarbeitung
            self.logger.info(f"Verarbeitete Daten:\nSpalten: {df.columns}\nIndex: {df.index.dtype}\nErste Zeilen:\n{df.head()}")
            
            self.logger.info(f"Daten geladen für {symbol}: {len(df)} Einträge von {df.index.min()} bis {df.index.max()}")
            
                return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten für {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_signals(self, df):
        """Berechnet die Handelssignale"""
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        for i in range(1, len(df)):
            # Kaufsignal: MA20 kreuzt MA50 nach oben und RSI < 70
            if (df['MA20'].iloc[i] > df['MA50'].iloc[i] and 
                df['MA20'].iloc[i-1] <= df['MA50'].iloc[i-1] and 
                df['RSI'].iloc[i] < 70):
                df.loc[df.index[i], 'buy_signal'] = True
            
            # Verkaufssignal: MA20 kreuzt MA50 nach unten oder RSI > 70
            elif (df['MA20'].iloc[i] < df['MA50'].iloc[i] and 
                  df['MA20'].iloc[i-1] >= df['MA50'].iloc[i-1] or 
                  df['RSI'].iloc[i] > 70):
                df.loc[df.index[i], 'sell_signal'] = True
        
        return df
    
    def _generate_analysis(self, df: pd.DataFrame) -> str:
        """
        Generiert eine formatierte Marktanalyse basierend auf den Daten.
        
        Args:
            df (pd.DataFrame): DataFrame mit den Aktiendaten
            
        Returns:
            str: Formatierte Marktanalyse
        """
        try:
            if df.empty:
                return "Keine Daten verfügbar für die Analyse."

            # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
            required_columns = ['Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Fehlende Spalten: {missing_cols}")

            # Aktuelle Werte als float konvertieren
            current_price = float(df['Close'].iloc[-1])
            previous_price = float(df['Close'].iloc[-2])
            price_change = ((current_price - previous_price) / previous_price) * 100
            
            current_volume = float(df['Volume'].iloc[-1])
            previous_volume = float(df['Volume'].iloc[-2])
            volume_change = ((current_volume - previous_volume) / previous_volume) * 100
            
            # Gleitende Durchschnitte berechnen
            ma20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
            ma50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
            ma200 = float(df['Close'].rolling(window=200).mean().iloc[-1])
            
            # RSI berechnen
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(rs.iloc[-1])
            rsi = 100 - (100 / (1 + rsi))

            # Analyse Text generieren
            analysis = f"""
### Marktstatistik
- Aktueller Preis: {current_price:.2f}
- Preisänderung: {price_change:.2f}%
- Volumenänderung: {volume_change:.2f}%

### Technische Indikatoren
- MA20: {ma20:.2f}
- MA50: {ma50:.2f}
- MA200: {ma200:.2f}
- RSI: {rsi:.2f}

### Marktbedingungen
"""
            # Trend Analyse
            if current_price > ma200:
                analysis += "- Langfristiger Aufwärtstrend\n"
            else:
                analysis += "- Langfristiger Abwärtstrend\n"

            if current_price > ma50:
                analysis += "- Mittelfristiger Aufwärtstrend\n"
            else:
                analysis += "- Mittelfristiger Abwärtstrend\n"

            if current_price > ma20:
                analysis += "- Kurzfristiger Aufwärtstrend\n"
            else:
                analysis += "- Kurzfristiger Abwärtstrend\n"

            # RSI Interpretation
            analysis += "\n### Handelsempfehlung\n"
            if rsi > 70:
                analysis += "- Überkauft (RSI > 70)\n"
            elif rsi < 30:
                analysis += "- Überverkauft (RSI < 30)\n"
            else:
                analysis += "- Neutrale RSI-Zone\n"

            return analysis

        except Exception as e:
            self.logger.error(f"Fehler bei der Marktanalyse-Generierung: {str(e)}")
            return "Fehler bei der Generierung der Marktanalyse."
    
    def _generate_lstm_analysis(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Generiert eine formatierte LSTM-Analyse basierend auf den Vorhersagen.
        
        Args:
            df (pd.DataFrame): DataFrame mit den LSTM-Vorhersagen
            symbol (str): Das Aktiensymbol
            
        Returns:
            str: Formatierte LSTM-Analysezusammenfassung
        """
        try:
            # Stelle sicher, dass die erforderlichen Spalten vorhanden sind
            required_columns = ['Close', 'LSTM_Prediction']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Fehlende Spalten für LSTM-Analyse: {missing_cols}")
                return f"Fehlende Daten für LSTM-Analyse: {', '.join(missing_cols)}"
            
            # Entferne NaN-Werte nur für die relevanten Spalten
            df_clean = df[required_columns].dropna()
            
            if df_clean.empty:
                return "Keine ausreichenden Daten für LSTM-Analyse verfügbar"
            
            # Aktuelle Werte als float konvertieren
            current_price = float(df_clean['Close'].iloc[-1])
            last_prediction = float(df_clean['LSTM_Prediction'].iloc[-1])
            
            # Berechne Vorhersage-Metriken
            pred_change = ((last_prediction - current_price) / current_price) * 100
            
            # Berechne Genauigkeitsmetriken
            close_values = df_clean['Close'].astype(float).values
            pred_values = df_clean['LSTM_Prediction'].astype(float).values
            
            mse = mean_squared_error(close_values, pred_values)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(close_values, pred_values))
            mape = float(np.mean(np.abs((close_values - pred_values) / close_values)) * 100)
            
            # Trendanalyse
            pred_series = df_clean['LSTM_Prediction'].astype(float)
            short_trend = pred_series.iloc[-5:].is_monotonic_increasing
            medium_trend = pred_series.iloc[-20:].is_monotonic_increasing
            
            analysis = f"""
## LSTM Vorhersage-Statistiken
- Aktueller Preis: ${current_price:.2f}
- Vorhergesagter Preis: ${last_prediction:.2f}
- Prognostizierte Änderung: {pred_change:.2f}%

## Modell-Metriken
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MAPE: {mape:.2f}%

## Trendanalyse
"""
            # Füge Trendanalyse hinzu
            if short_trend and medium_trend:
                analysis += "- Starker Aufwärtstrend in allen Zeitfenstern\n"
            elif short_trend:
                analysis += "- Kurzfristiger Aufwärtstrend erkennbar\n"
            elif medium_trend:
                analysis += "- Mittelfristiger Aufwärtstrend erkennbar\n"
            else:
                analysis += "- Kein klarer Trend erkennbar\n"
            
            analysis += "\n## Handelsempfehlung basierend auf LSTM\n"
            if pred_change > 2:
                analysis += "- Starkes Kaufsignal (>2% erwartete Rendite)\n"
            elif pred_change > 0.5:
                analysis += "- Moderates Kaufsignal\n"
            elif pred_change < -2:
                analysis += "- Starkes Verkaufssignal\n"
            elif pred_change < -0.5:
                analysis += "- Moderates Verkaufssignal\n"
            else:
                analysis += "- Neutral - Seitwärtsbewegung erwartet\n"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler bei der LSTM-Analyse-Generierung: {str(e)}")
            return "Fehler bei der Generierung der LSTM-Analyse. Bitte überprüfen Sie die Logs."
    
    def _create_price_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Erstellt ein Candlestick-Chart mit technischen Indikatoren.
        
        Args:
            df: DataFrame mit den Aktiendaten
            symbol: Das Aktiensymbol für den Titel
            
        Returns:
            go.Figure: Plotly Figure-Objekt mit dem Chart
        """
        try:
            # Debug-Log für Eingangsdaten
            self.logger.info(f"Erstelle Chart für {symbol}")
            self.logger.info(f"Verfügbare Spalten: {df.columns.tolist()}")
            self.logger.info(f"Erste Zeilen der Daten:\n{df.head()}")
            
            # Erstelle Figure mit Subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Aktienkurs', 'Volumen'),
                row_heights=[0.7, 0.3]
            )

            # Candlestick Chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Füge Moving Averages hinzu
            colors = {'MA20': '#1f77b4', 'MA50': '#ff7f0e', 'MA200': '#2ca02c'}
            for ma in ['MA20', 'MA50', 'MA200']:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ma],
                            name=ma,
                            line=dict(color=colors[ma]),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )

            # Volumen-Balken
            colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )

            # Layout anpassen
            fig.update_layout(
                title=f'{symbol} Technische Analyse',
                xaxis_title='Datum',
                yaxis_title='Preis',
                yaxis2_title='Volumen',
                showlegend=True,
                height=800,
                xaxis_rangeslider_visible=False
            )

            # Formatierung der y-Achsen
            fig.update_yaxes(title_text='Preis', row=1, col=1)
            fig.update_yaxes(title_text='Volumen', row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Charts für {symbol}: {str(e)}")
            # Erstelle leeren Chart bei Fehler
            return go.Figure()
    
    def _create_indicators_chart(self, df, symbol):
        """Erstellt den Indikatoren-Chart mit RSI
        
        Args:
            df (pd.DataFrame): DataFrame mit den Kursdaten
            symbol (str): Das Aktiensymbol
            
        Returns:
            go.Figure: Plotly Figure-Objekt mit dem Chart
        """
        try:
            if df.empty:
                return self._create_error_chart()
                
            # Stelle sicher, dass das DataFrame sortiert ist
            df = df.sort_index()
            
            # Berechne RSI falls noch nicht vorhanden
            if 'RSI' not in df.columns:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            # Erstelle Figure
            fig = go.Figure()
            
            # RSI
                fig.add_trace(go.Scatter(
                x=df.index,
                    y=df['RSI'],
                    name='RSI',
                line=dict(color='purple', width=1.5)
                ))
                
                # RSI-Linien
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Überkauft (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Überverkauft (30)")
            
            # Layout
            fig.update_layout(
                title=f'{symbol} Technische Indikatoren',
                yaxis_title='RSI',
                xaxis_title='Datum',
                height=300,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Zeitachse konfigurieren
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1J", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Y-Achse auf RSI-Bereich beschränken
            fig.update_yaxes(range=[0, 100])
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Indikatoren-Charts: {str(e)}")
            return self._create_error_chart()
    
    def _create_error_chart(self):
        """Erstellt einen Error-Chart"""
        fig = go.Figure()
        fig.add_annotation(
            text="Keine Daten verfügbar",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(template='plotly_white')
        return fig
    
    def _load_metrics(self, symbol):
        """Lädt die Performance-Metriken"""
        try:
            file_path = Path(f"ml4t_project/exports/metrics_{symbol}.json")
            if file_path.exists():
                import json
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                return html.Div([
                    html.P(f"Genauigkeit: {metrics.get('accuracy', 'N/A')}%"),
                    html.P(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}"),
                    html.P(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
                ])
            return "Keine Metriken verfügbar"
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Metriken: {str(e)}")
            return "Fehler beim Laden der Metriken"
    
    def _load_signals(self, symbol):
        """Lädt die aktuellen Handelssignale"""
        try:
            file_path = Path(f"ml4t_project/exports/signals_{symbol}.json")
            if file_path.exists():
                import json
                with open(file_path, 'r') as f:
                    signals = json.load(f)
                return html.Div([
                    html.P(f"Aktuelles Signal: {signals.get('current_signal', 'N/A')}"),
                    html.P(f"Signalstärke: {signals.get('signal_strength', 'N/A')}"),
                    html.P(f"Letztes Update: {signals.get('last_update', 'N/A')}")
                ])
            return "Keine Signale verfügbar"
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Signale: {str(e)}")
            return "Fehler beim Laden der Signale"
    
    def _generate_lstm_predictions(self, df):
        """Generiert LSTM-Vorhersagen"""
        try:
            # Implementiere die Logik zur Generierung von LSTM-Vorhersagen
            # Dies ist nur ein Beispiel und sollte an Ihre spezifischen Anforderungen angepasst werden
            # Hier wird eine einfache Vorhersage basierend auf der letzten Preisänderung verwendet
            last_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            return np.array([df['Close'].iloc[-1] + last_change] * 60)
        except Exception as e:
            self.logger.error(f"Fehler beim Generieren der LSTM-Vorhersagen: {str(e)}")
            return np.array([np.nan] * 60)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für die Analyse.
        
        Args:
            df: DataFrame mit den Kursdaten
            
        Returns:
            pd.DataFrame: DataFrame mit hinzugefügten technischen Indikatoren
        """
        try:
            # Moving Averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # Handelssignale
            df['buy_signal'] = False
            df['sell_signal'] = False
            
            # Kaufsignal: Kurs kreuzt MA50 von unten nach oben
            df['buy_signal'] = (df['Close'] > df['MA50']) & (df['Close'].shift(1) <= df['MA50'].shift(1))
            
            # Verkaufssignal: Kurs kreuzt MA50 von oben nach unten
            df['sell_signal'] = (df['Close'] < df['MA50']) & (df['Close'].shift(1) >= df['MA50'].shift(1))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der technischen Indikatoren: {str(e)}")
            return df
    
    def run(self, debug=False, port=8050):
        """Startet das Dashboard"""
        try:
            super().run(debug=debug, port=port)
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")

if __name__ == '__main__':
    dashboard = MonitoringDashboard()
    dashboard.run(debug=True) 