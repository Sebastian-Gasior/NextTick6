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
from datetime import datetime
import sys

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ml4t_project.analysis.market_analyzer import MarketAnalyzer
from ml4t_project.visualization.show_chart import create_chart_for_dashboard, save_chart_to_file

class MonitoringDashboard:
    """Trading-Dashboard mit Aktienanalyse und Handelssignalen"""
    
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            assets_folder=os.path.join(os.getcwd(), "ml4t_project/exports")
        )
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Analyzer
        self.market_analyzer = MarketAnalyzer()
        
        # Standard-Zeitraum für Daten
        self.start_date = "2022-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Logging Setup
        logging.basicConfig(
            filename=self.log_dir / "trading.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Layout definieren
        self.app.layout = html.Div([
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
            
            # Hauptbereich - 2 Spalten Layout
            html.Div([
                # Linke Spalte - Charts
                html.Div([
                    # Kursverlauf
                    html.Div([
                        html.H3("Kursverlauf und Signale"),
                        dcc.Graph(id='price-chart')
                    ], style={'marginBottom': '20px'}),
                    
                    # LSTM-Vorhersage Chart
                    html.Div([
                        html.H3("LSTM-Vorhersage"),
                        dcc.Graph(id='lstm-prediction-chart')
                    ], style={'marginBottom': '20px'}),
                    
                    # Technische Indikatoren
                    html.Div([
                        html.H3("Technische Indikatoren"),
                        dcc.Graph(id='indicators-chart')
                    ])
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Rechte Spalte - Analysen und Metriken
                html.Div([
                    # Marktanalyse
                    html.Div([
                        html.H3("Marktanalyse"),
                        html.Pre(id='market-analysis', 
                               style={'padding': '10px', 'border': '1px solid #ddd', 'whiteSpace': 'pre-wrap',
                                     'fontFamily': 'monospace', 'fontSize': '14px', 'backgroundColor': '#f8f9fa',
                                     'overflowY': 'auto', 'maxHeight': '500px'})
                    ], style={'marginBottom': '20px'}),
                    
                    # Performance-Metriken
                    html.Div([
                        html.H3("Performance-Metriken"),
                        html.Div(id='performance-metrics',
                               style={'padding': '10px', 'border': '1px solid #ddd'})
                    ], style={'marginBottom': '20px'}),
                    
                    # Handelssignale
                    html.Div([
                        html.H3("Aktuelle Handelssignale"),
                        html.Div(id='trading-signals',
                               style={'padding': '10px', 'border': '1px solid #ddd'})
                    ]),
                    
                    # Detaillierte LSTM-Vorhersage
                    html.Div([
                        html.H3("Detaillierte LSTM-Vorhersage"),
                        html.Iframe(
                            id='lstm-iframe',
                            style={'width': '100%', 'height': '300px', 'border': '1px solid #ddd'}
                        )
                    ], style={'marginTop': '20px'})
                ], style={'width': '28%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
            ], style={'margin': '20px'})
        ], style={'maxWidth': '1800px', 'margin': 'auto', 'padding': '20px'})
        
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('lstm-prediction-chart', 'figure'),
             Output('market-analysis', 'children'),
             Output('performance-metrics', 'children'),
             Output('trading-signals', 'children')],
            [Input('stock-selector', 'value')]
        )
        def update_charts(symbol):
            try:
                self.logger.info(f"Aktualisiere Charts für {symbol}")
                
                # Lade Daten
                df = self._load_data(symbol)
                if df is None or df.empty:
                    raise ValueError(f"Keine Daten für {symbol} verfügbar")
                
                # Berechne Handelssignale
                df = self._calculate_signals(df)
                
                # Erstelle Charts
                price_fig = self._create_price_chart(df, symbol)
                indicators_fig = self._create_indicators_chart(df, symbol)
                
                # Erstelle LSTM-Vorhersage Chart
                lstm_fig = create_chart_for_dashboard(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                # Generiere Analyse
                analysis = self._generate_analysis(df, symbol)
                self.logger.info(f"Analyse für {symbol} generiert")
                
                # Lade Metriken und Signale
                metrics = self._load_metrics(symbol)
                signals = self._load_signals(symbol)
                
                # Speichere detaillierten LSTM-Chart
                try:
                    save_chart_to_file(
                        symbol=symbol, 
                        start_date=self.start_date, 
                        end_date=self.end_date
                    )
                except Exception as e:
                    self.logger.error(f"Fehler beim Speichern des LSTM-Charts: {str(e)}")
                
                return price_fig, indicators_fig, lstm_fig, analysis, metrics, signals
                
            except Exception as e:
                self.logger.error(f"Fehler beim Update der Charts: {str(e)}")
                return self._create_error_chart(), self._create_error_chart(), \
                       self._create_error_chart(), \
                       f"Fehler bei der Marktanalyse für {symbol}: {str(e)}", \
                       "Fehler beim Laden der Metriken", \
                       "Fehler beim Laden der Signale"
                       
        @self.app.callback(
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
    
    def _load_data(self, symbol):
        """Lädt die Daten für das ausgewählte Symbol"""
        try:
            file_path = Path(f"ml4t_project/exports/data_{symbol}.csv")
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten: {str(e)}")
            return None
    
    def _calculate_signals(self, df):
        """Berechnet die Handelssignale"""
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        for i in range(1, len(df)):
            # Kaufsignal: MA20 kreuzt MA50 nach oben und RSI < 70
            if (df['SMA20'].iloc[i] > df['SMA50'].iloc[i] and 
                df['SMA20'].iloc[i-1] <= df['SMA50'].iloc[i-1] and 
                df['RSI'].iloc[i] < 70):
                df.loc[df.index[i], 'buy_signal'] = True
            
            # Verkaufssignal: MA20 kreuzt MA50 nach unten oder RSI > 70
            elif (df['SMA20'].iloc[i] < df['SMA50'].iloc[i] and 
                  df['SMA20'].iloc[i-1] >= df['SMA50'].iloc[i-1] or 
                  df['RSI'].iloc[i] > 70):
                df.loc[df.index[i], 'sell_signal'] = True
        
        return df
    
    def _generate_analysis(self, df: pd.DataFrame, symbol: str) -> str:
        """Generiert eine detaillierte Marktanalyse"""
        try:
            # Grundlegende Statistiken
            stats = {
                'datenpunkte': len(df),
                'zeitraum': {
                    'start': df['Date'].iloc[0].strftime('%Y-%m-%d'),
                    'end': df['Date'].iloc[-1].strftime('%Y-%m-%d')
                },
                'preisbereich': {
                    'min': df['Low'].min(),
                    'max': df['High'].max()
                }
            }
            
            # Aktuelle Werte
            current_price = df['Close'].iloc[-1]
            current_ma20 = df['SMA20'].iloc[-1]
            current_ma50 = df['SMA50'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            # Performance berechnen
            returns = df['Close'].pct_change()
            total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
            
            # Signale identifizieren
            buy_signals = []
            sell_signals = []
            in_position = False
            
            for i in range(1, len(df)):
                if df['SMA20'].iloc[i] > df['SMA50'].iloc[i] and df['SMA20'].iloc[i-1] <= df['SMA50'].iloc[i-1]:
                    if df['RSI'].iloc[i] < 70:
                        buy_signals.append({
                            'datum': df['Date'].iloc[i].strftime('%Y-%m-%d'),
                            'preis': df['Close'].iloc[i]
                        })
                        in_position = True
                elif in_position and (df['SMA20'].iloc[i] < df['SMA50'].iloc[i] or df['RSI'].iloc[i] > 70):
                    sell_signals.append({
                        'datum': df['Date'].iloc[i].strftime('%Y-%m-%d'),
                        'preis': df['Close'].iloc[i]
                    })
                    in_position = False
            
            # Generiere Analysebericht
            analysis = f"""
Marktanalyse für {symbol}

1. Grundlegende Statistiken:
   Datenpunkte: {stats['datenpunkte']} ({stats['zeitraum']['start']} bis {stats['zeitraum']['end']})
   Preisbereich: ${stats['preisbereich']['min']:.2f} - ${stats['preisbereich']['max']:.2f}

2. Performance:
   Gesamtrendite: {total_return:.1f}%
   Volatilität (annualisiert): {volatility:.1f}%
   Sharpe Ratio: {sharpe:.2f}
   Max Drawdown: {max_drawdown:.1f}%

3. Handelssignale:
   Anzahl Kaufsignale: {len(buy_signals)}
   Anzahl Verkaufssignale: {len(sell_signals)}

   Letzte Signale:"""
            
            # Füge die letzten 3 Signale hinzu
            all_signals = []
            for buy, sell in zip(buy_signals, sell_signals):
                perf = ((sell['preis'] - buy['preis']) / buy['preis']) * 100
                all_signals.append(f"   Signal: {buy['datum']} Kauf @${buy['preis']:.2f}, {sell['datum']} Verkauf @${sell['preis']:.2f} ({perf:+.1f}%)")
            
            if all_signals:
                analysis += "\n" + "\n".join(all_signals[-3:])
            
            analysis += f"""

4. Aktuelle Marktsituation:
   Preis: ${current_price:.2f}
   MA20: {current_ma20:.2f}
   MA50: {current_ma50:.2f}
   RSI: {current_rsi:.2f}
   
   Trend: {"Aufwärts" if current_ma20 > current_ma50 else "Abwärts"}
   RSI-Status: {"Überkauft" if current_rsi > 70 else "Überverkauft" if current_rsi < 30 else "Neutral"}

5. Handlungsempfehlung:"""
            
            # Generiere Handlungsempfehlung
            if current_ma20 > current_ma50 and current_rsi < 70:
                analysis += """
   KAUFEN: Positiver Trend mit moderatem RSI
   - MA20 über MA50 deutet auf Aufwärtstrend
   - RSI zeigt noch keine überkaufte Situation
   - LSTM-Vorhersage zeigt möglichen weiteren Anstieg"""
            elif current_ma20 < current_ma50 and current_rsi > 30:
                analysis += """
   VERKAUFEN: Negativer Trend, kein überverkaufter RSI
   - MA20 unter MA50 deutet auf Abwärtstrend
   - RSI zeigt keine überverkaufte Situation
   - LSTM-Vorhersage deutet auf mögliche Fortsetzung des Abwärtstrends"""
            else:
                analysis += """
   HALTEN: Keine klaren Signale
   - Warten auf eindeutige Trendbestätigung
   - Beobachten der RSI-Entwicklung
   - LSTM-Vorhersage berücksichtigen für kurzfristige Bewegungen"""
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse-Generierung: {str(e)}")
            raise
    
    def _create_price_chart(self, df, symbol):
        """Erstellt den Kursverlauf-Chart mit Handelssignalen"""
        try:
            fig = go.Figure()
            
            # Kerzen-Chart
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Kurs',
                showlegend=True
            ))
            
            # Moving Averages
            if 'SMA20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['SMA20'],
                    name='20-Tage MA',
                    line=dict(color='blue'),
                    showlegend=True
                ))
            
            if 'SMA50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['SMA50'],
                    name='50-Tage MA',
                    line=dict(color='red'),
                    showlegend=True
                ))
            
            # Kaufsignale
            buy_signals = df[df['buy_signal']]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Low'] * 0.99,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Kaufsignal',
                    showlegend=True
                ))
            
            # Verkaufssignale
            sell_signals = df[df['sell_signal']]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['High'] * 1.01,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Verkaufssignal',
                    showlegend=True
                ))
            
            # Layout
            fig.update_layout(
                title=f'{symbol} Kursverlauf mit Handelssignalen',
                yaxis_title='Preis',
                xaxis_title='Datum',
                height=500,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02
                )
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Preis-Charts: {str(e)}")
            return self._create_error_chart()
    
    def _create_indicators_chart(self, df, symbol):
        """Erstellt den Indikatoren-Chart"""
        try:
            fig = go.Figure()
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ))
                
                # RSI-Linien
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                fig.add_hline(y=30, line_dash="dash", line_color="green")
            
            # Layout
            fig.update_layout(
                title=f'{symbol} Technische Indikatoren',
                yaxis_title='Wert',
                xaxis_title='Datum',
                height=300,
                template='plotly_white'
            )
            
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
    
    def run(self, debug=False, port=8050):
        """Startet das Dashboard"""
        try:
            self.app.run(debug=debug, port=port)
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")

if __name__ == '__main__':
    dashboard = MonitoringDashboard()
    dashboard.run(debug=True) 