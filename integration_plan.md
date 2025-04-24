# Integrationsplan: master_analysis.py und show_chart.py

## Aktuelle Situation

1. **master_analysis.py**:
   - Hauptskript für die ML4T Trading-Analyse
   - Initialisiert die `TradingAnalysis`-Klasse
   - Lädt Marktdaten über die `MarketData`-Klasse
   - Führt Analysen mit der `MarketAnalyzer`-Klasse durch
   - Startet ein Dashboard über die `MonitoringDashboard`-Klasse

2. **show_chart.py**:
   - Visualisierungskomponente im Verzeichnis `ml4t_project/visualization/`
   - Generiert interaktive Charts mit LSTM-Vorhersagen
   - Lädt Aktiendaten über `yfinance`
   - Erstellt und exportiert HTML-Charts
   - Wird aktuell nicht in `master_analysis.py` verwendet

3. **Problematik**:
   - Die HTML-Charts aus `show_chart.py` werden nicht im Dashboard angezeigt
   - Die LSTM-Vorhersagen werden nicht in das Dashboard integriert
   - Es gibt keine direkte Verbindung zwischen beiden Komponenten

## Integrationsplan

### 1. Modifikation von show_chart.py

```python
# ml4t_project/visualization/show_chart.py (Änderungen)

# Funktion für direkten Dashboard-Zugriff hinzufügen
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
        # Daten laden (falls nicht übergeben)
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} gefunden")
            
        # Verarbeitung wie in main()
        # ...
        
        # LSTM-Vorhersagen verwenden oder generieren
        if lstm_predictions is not None:
            predictions = lstm_predictions
        else:
            predictions = get_lstm_predictions(df)
            
        # Füge Vorhersagen zum DataFrame hinzu
        df['LSTM_Prediction'] = np.nan
        df.iloc[60:len(predictions)+60, df.columns.get_loc('LSTM_Prediction')] = predictions.flatten()
        
        # Erstelle Figure (wie in main())
        fig = go.Figure()
        # ... (Chart-Erstellung)
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Chart-Erstellung: {str(e)}")
        raise
```

### 2. Anpassung des MonitoringDashboard

```python
# ml4t_project/monitoring/dashboard.py (Änderungen)

from ml4t_project.visualization.show_chart import create_chart_for_dashboard

class MonitoringDashboard:
    # ...
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('lstm-prediction-chart', 'figure'),  # Neuer Output
             Output('market-analysis', 'children'),
             Output('performance-metrics', 'children'),
             Output('trading-signals', 'children')],
            [Input('stock-selector', 'value')]
        )
        def update_charts(symbol):
            try:
                # ... (bestehender Code)
                
                # LSTM-Vorhersage-Chart erstellen
                lstm_fig = create_chart_for_dashboard(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                return price_fig, indicators_fig, lstm_fig, analysis, metrics, signals
                
            except Exception as e:
                # ... (Fehlerbehandlung)
                return self._create_error_chart(), self._create_error_chart(), \
                       self._create_error_chart(), \
                       f"Fehler bei der Marktanalyse für {symbol}: {str(e)}", \
                       "Fehler beim Laden der Metriken", \
                       "Fehler beim Laden der Signale"
```

### 3. Anpassung des Dashboard-Layouts

```python
# ml4t_project/monitoring/dashboard.py (Layout-Änderungen)

# Zum Layout hinzufügen:
html.Div([
    html.H3("LSTM-Vorhersage"),
    dcc.Graph(id='lstm-prediction-chart')
], style={'marginBottom': '20px'})
```

### 4. Integration in TradingAnalysis

```python
# master_analysis.py (Änderungen)

from ml4t_project.visualization.show_chart import get_lstm_predictions

class TradingAnalysis:
    # ...
    
    def prepare_data(self):
        """Bereitet die Daten für alle Symbole vor"""
        logger.info("Lade Marktdaten...")
        
        for symbol in self.symbols:
            try:
                # ... (bestehender Code)
                
                if df is not None:
                    # ... (bestehender Code)
                    
                    # Generiere LSTM-Vorhersagen
                    try:
                        predictions = get_lstm_predictions(df)
                        
                        # Speichere Vorhersagen
                        predictions_file = Path(f"ml4t_project/exports/lstm_predictions_{symbol}.npy")
                        np.save(predictions_file, predictions)
                        logger.info(f"LSTM-Vorhersagen für {symbol} generiert und gespeichert")
                    except Exception as e:
                        logger.error(f"Fehler bei LSTM-Vorhersagen für {symbol}: {str(e)}")
                        
            except Exception as e:
                # ... (Fehlerbehandlung)
```

### 5. Dateisystemintegration für vorberechnete Charts

```python
# ml4t_project/visualization/show_chart.py (zusätzliche Funktion)

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
        
        return output_file
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Charts: {str(e)}")
        raise
```

### 6. Integration in das Dashboard über iframe

```python
# ml4t_project/monitoring/dashboard.py (optionale iframe-Integration)

# Zum Dashboard-Layout hinzufügen:
html.Div([
    html.H3("Detaillierte LSTM-Vorhersage"),
    html.Iframe(
        id='lstm-iframe',
        style={'width': '100%', 'height': '600px', 'border': 'none'}
    )
])

# Callback hinzufügen:
@self.app.callback(
    Output('lstm-iframe', 'src'),
    [Input('stock-selector', 'value')]
)
def update_iframe(symbol):
    # Prüfe, ob Chart existiert, sonst generiere ihn
    chart_path = f"ml4t_project/exports/charts/chart_{symbol}.html"
    if not os.path.exists(chart_path):
        from ml4t_project.visualization.show_chart import save_chart_to_file
        try:
            save_chart_to_file(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Charts: {str(e)}")
            return ""
    
    # Relativen Pfad für iframe zurückgeben
    return f"/assets/charts/chart_{symbol}.html"
```

### 7. Anpassung der Verzeichniskonfiguration für Assets

```python
# ml4t_project/monitoring/dashboard.py (Asset-Konfiguration)

# Beim Initialisieren des Dash-App:
self.app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.getcwd(), "ml4t_project/exports")
)
```

## Implementierungsreihenfolge

1. Modifiziere `show_chart.py` und füge die neue Funktion `create_chart_for_dashboard` hinzu
2. Aktualisiere `save_chart_to_file` um vorgefertigte Charts zu speichern
3. Aktualisiere das Dashboard-Layout mit einem zusätzlichen Tab/Bereich für LSTM-Vorhersagen
4. Integriere die neue Funktion in die Callbacks des Dashboards
5. Modifiziere `TradingAnalysis` um LSTM-Vorhersagen bei der Datenvorbereitung zu generieren
6. Füge iframe-Integration für detaillierte Charts hinzu
7. Konfiguriere das Asset-Verzeichnis für die Web-App

## Testplan

1. Teste die neue `create_chart_for_dashboard`-Funktion isoliert
2. Teste die Generierung und Speicherung der Charts
3. Teste die Dashboard-Integration mit einfachen Daten
4. Überprüfe die korrekte Anzeige der LSTM-Vorhersagen
5. Teste die Integration mit realen Daten
6. Überprüfe die iframe-Integration und Zugriff auf gespeicherte Charts
7. Führe einen End-to-End-Test mit dem vollständigen Workflow durch

## Konfigurationsanpassungen

1. Stelle sicher, dass das Verzeichnis `ml4t_project/exports/charts` existiert
2. Überprüfe die Konfiguration in `trading_config.yaml`
3. Füge die notwendigen Abhängigkeiten hinzu (falls erforderlich)
4. Aktualisiere die Dokumentation 