"""
Erweiterte Marktanalyse für ML4T
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class MarketAnalyzer:
    """Generiert detaillierte Marktanalysen für Aktien"""
    
    def __init__(self):
        self.exports_dir = Path("ml4t_project/exports")
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Stelle sicher, dass das Analyse-Verzeichnis existiert
        self.analysis_dir = self.exports_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_stock(self, df: pd.DataFrame, symbol: str) -> str:
        """Generiert eine detaillierte Marktanalyse"""
        try:
            self.logger.info(f"Starte Analyse für {symbol}")
            
            # Grundlegende Statistiken
            stats = self._calculate_basic_stats(df)
            self.logger.info(f"Grundlegende Statistiken berechnet: {len(df)} Datenpunkte")
            
            # Handelssignale identifizieren
            signals = self._identify_trading_signals(df)
            self.logger.info(f"Handelssignale identifiziert: {len(signals)} Signale")
            
            # Performance der Signale berechnen
            performance = self._calculate_signal_performance(df, signals)
            self.logger.info(f"Performance berechnet: {len(performance.get('trades', []))} Trades")
            
            # MA-Analyse für spezifische Zeiträume
            ma_analysis = self._analyze_moving_averages(df)
            
            # Generiere Analysebericht
            analysis = self._generate_analysis_report(
                symbol, stats, signals, performance, ma_analysis, df
            )
            
            # Speichere Analyse
            self._save_analysis(symbol, analysis)
            self.logger.info(f"Analyse für {symbol} erfolgreich gespeichert")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Marktanalyse für {symbol}: {str(e)}")
            return f"Fehler bei der Marktanalyse für {symbol}: {str(e)}"
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Berechnet grundlegende Statistiken"""
        try:
            return {
                'datenpunkte': len(df),
                'zeitraum': {
                    'start': df['Date'].iloc[0].strftime('%Y-%m-%d'),
                    'end': df['Date'].iloc[-1].strftime('%Y-%m-%d')
                },
                'preisbereich': {
                    'min': df['Low'].min(),
                    'max': df['High'].max()
                },
                'volumen': {
                    'durchschnitt': df['Volume'].mean(),
                    'max': df['Volume'].max()
                }
            }
        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der Statistiken: {str(e)}")
            raise
    
    def _identify_trading_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Identifiziert Handelssignale basierend auf MA-Kreuzungen und RSI"""
        try:
            signals = []
            
            # Initialisiere Signal-Status
            in_position = False
            entry_price = 0
            entry_date = None
            
            for i in range(1, len(df)):
                date = df['Date'].iloc[i]
                price = df['Close'].iloc[i]
                ma20 = df['SMA20'].iloc[i]
                ma50 = df['SMA50'].iloc[i]
                rsi = df['RSI'].iloc[i]
                prev_ma20 = df['SMA20'].iloc[i-1]
                prev_ma50 = df['SMA50'].iloc[i-1]
                
                # Kaufsignal
                if not in_position and prev_ma20 <= prev_ma50 and ma20 > ma50 and rsi < 70:
                    signals.append({
                        'typ': 'KAUF',
                        'datum': date,
                        'preis': price,
                        'grund': f"MA20 kreuzt MA50 nach oben, RSI = {rsi:.2f}"
                    })
                    in_position = True
                    entry_price = price
                    entry_date = date
                
                # Verkaufssignal
                elif in_position and (prev_ma20 >= prev_ma50 and ma20 < ma50 or rsi > 70):
                    performance = ((price - entry_price) / entry_price) * 100
                    signals.append({
                        'typ': 'VERKAUF',
                        'datum': date,
                        'preis': price,
                        'performance': performance,
                        'grund': f"MA20 kreuzt MA50 nach unten oder RSI überkauft ({rsi:.2f})"
                    })
                    in_position = False
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Signalidentifikation: {str(e)}")
            raise
    
    def _calculate_signal_performance(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """Berechnet die Performance der Handelssignale"""
        try:
            performance = {
                'anzahl_kauf': len([s for s in signals if s['typ'] == 'KAUF']),
                'anzahl_verkauf': len([s for s in signals if s['typ'] == 'VERKAUF']),
                'trades': []
            }
            
            # Paare Kauf- und Verkaufssignale
            i = 0
            while i < len(signals) - 1:
                if signals[i]['typ'] == 'KAUF':
                    kauf = signals[i]
                    verkauf = signals[i + 1]
                    perf = ((verkauf['preis'] - kauf['preis']) / kauf['preis']) * 100
                    
                    performance['trades'].append({
                        'kauf_datum': kauf['datum'].strftime('%Y-%m-%d'),
                        'kauf_preis': kauf['preis'],
                        'verkauf_datum': verkauf['datum'].strftime('%Y-%m-%d'),
                        'verkauf_preis': verkauf['preis'],
                        'performance': perf
                    })
                    i += 2
                else:
                    i += 1
            
            if performance['trades']:
                performance['beste_performance'] = max([t['performance'] for t in performance['trades']])
                performance['schlechteste_performance'] = min([t['performance'] for t in performance['trades']])
                performance['durchschnitt_performance'] = np.mean([t['performance'] for t in performance['trades']])
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Performance-Berechnung: {str(e)}")
            raise
    
    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Analysiert die Moving Averages für bestimmte Zeiträume"""
        try:
            analysis = {}
            
            # Analysiere Januar 2023
            jan_2023 = df[df['Date'].dt.month == 1]
            if not jan_2023.empty:
                analysis['januar_2023'] = {
                    'start': {
                        'datum': jan_2023['Date'].iloc[0].strftime('%Y-%m-%d'),
                        'ma20': jan_2023['SMA20'].iloc[0],
                        'ma50': jan_2023['SMA50'].iloc[0],
                        'rsi': jan_2023['RSI'].iloc[0]
                    },
                    'ende': {
                        'datum': jan_2023['Date'].iloc[-1].strftime('%Y-%m-%d'),
                        'ma20': jan_2023['SMA20'].iloc[-1],
                        'ma50': jan_2023['SMA50'].iloc[-1],
                        'rsi': jan_2023['RSI'].iloc[-1]
                    }
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler bei der MA-Analyse: {str(e)}")
            raise
    
    def _generate_analysis_report(self, symbol: str, stats: Dict, signals: List[Dict], 
                                performance: Dict, ma_analysis: Dict, df: pd.DataFrame) -> str:
        """Generiert den vollständigen Analysebericht"""
        try:
            report = f"""
Marktanalyse für {symbol}

1. Grundlegende Statistiken:
   Datenpunkte: {stats['datenpunkte']} ({stats['zeitraum']['start']} bis {stats['zeitraum']['end']})
   Preisbereich: ${stats['preisbereich']['min']:.2f} - ${stats['preisbereich']['max']:.2f}
   Durchschnittliches Volumen: {stats['volumen']['durchschnitt']:,.0f}

2. Handelssignale:
   Anzahl Kaufsignale: {performance['anzahl_kauf']}
   Anzahl Verkaufssignale: {performance['anzahl_verkauf']}

3. Performance der Trades:"""

            if performance['trades']:
                for i, trade in enumerate(performance['trades'], 1):
                    report += f"""
   Signal {i}: {trade['performance']:.1f}% (Kauf @${trade['kauf_preis']:.2f}, Verkauf @${trade['verkauf_preis']:.2f})"""
                
                report += f"""
   
   Beste Performance: {performance['beste_performance']:.1f}%
   Schlechteste Performance: {performance['schlechteste_performance']:.1f}%
   Durchschnittliche Performance: {performance['durchschnitt_performance']:.1f}%"""
            
            if 'januar_2023' in ma_analysis:
                jan = ma_analysis['januar_2023']
                report += f"""

4. Detaillierte MA-Analyse (Januar 2023):
   Anfang des Monats ({jan['start']['datum']}):
   - MA20: {jan['start']['ma20']:.2f}
   - MA50: {jan['start']['ma50']:.2f}
   - RSI: {jan['start']['rsi']:.2f}
   
   Ende des Monats ({jan['ende']['datum']}):
   - MA20: {jan['ende']['ma20']:.2f}
   - MA50: {jan['ende']['ma50']:.2f}
   - RSI: {jan['ende']['rsi']:.2f}"""
            
            # Aktuelle Marktsituation
            current_price = df['Close'].iloc[-1]
            current_ma20 = df['SMA20'].iloc[-1]
            current_ma50 = df['SMA50'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            report += f"""

5. Aktuelle Marktsituation:
   Preis: ${current_price:.2f}
   MA20: {current_ma20:.2f}
   MA50: {current_ma50:.2f}
   RSI: {current_rsi:.2f}
   
   Trend: {"Aufwärts" if current_ma20 > current_ma50 else "Abwärts"}
   RSI-Status: {"Überkauft" if current_rsi > 70 else "Überverkauft" if current_rsi < 30 else "Neutral"}"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Report-Generierung: {str(e)}")
            raise
    
    def _save_analysis(self, symbol: str, analysis: str):
        """Speichert die Analyse als Textdatei"""
        try:
            file_path = self.analysis_dir / f"analysis_{symbol}.txt"
            
            # Stelle sicher, dass das Verzeichnis existiert
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Speichere Analyse in: {file_path}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
                
            self.logger.info(f"Analyse erfolgreich gespeichert für {symbol}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Analyse: {str(e)}")
            raise 