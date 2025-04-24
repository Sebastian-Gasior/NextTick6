"""
Textuelle Analyse-Generator für ML4T
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

class TextAnalyzer:
    """Generiert detaillierte textuelle Analysen für Aktien"""
    
    def analyze_stock_movement(self, df: pd.DataFrame, predictions: pd.Series) -> str:
        """Analysiert die Aktienbewegung und generiert eine detaillierte Beschreibung"""
        
        # Berechne wichtige Metriken
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = (current_price - prev_price) / prev_price * 100
        
        volume_change = (df['Volume'].iloc[-1] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2] * 100
        
        # Trend-Analyse
        sma20 = df['Close'].rolling(window=20).mean()
        sma50 = df['Close'].rolling(window=50).mean()
        
        trend = "aufwärts" if sma20.iloc[-1] > sma50.iloc[-1] else "abwärts"
        trend_strength = abs(sma20.iloc[-1] - sma50.iloc[-1]) / current_price * 100
        
        # Volatilitäts-Analyse
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # Momentum-Analyse
        rsi = self._calculate_rsi(df['Close'])
        momentum = "stark positiv" if rsi > 70 else "stark negativ" if rsi < 30 else "neutral"
        
        # Vorhersage-Analyse
        pred_change = (predictions.iloc[-1] - current_price) / current_price * 100
        pred_direction = "steigend" if pred_change > 0 else "fallend"
        
        # Generiere Text
        analysis = f"""
Aktuelle Marktanalyse:

1. Kursentwicklung:
   - Aktueller Kurs: ${current_price:.2f}
   - Tagesveränderung: {price_change:+.2f}%
   - Handelsvolumen: {volume_change:+.2f}% zum Vortag

2. Technische Analyse:
   - Trend: {trend.capitalize()}trend mit {trend_strength:.1f}% Stärke
   - Volatilität: {volatility:.1f}% (annualisiert)
   - Momentum: {momentum.capitalize()}
   - RSI: {rsi:.1f}

3. Vorhersage:
   - Richtung: {pred_direction.capitalize()}
   - Erwartete Änderung: {pred_change:+.2f}%
   - Konfidenz: {self._calculate_confidence(pred_change, volatility):.1f}%

4. Handelsempfehlung:
{self._generate_trading_recommendation(price_change, pred_change, rsi, trend)}

5. Risikobewertung:
{self._assess_risk(volatility, volume_change, trend_strength)}
"""
        
        return analysis
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Berechnet den RSI"""
        
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_confidence(self, pred_change: float, volatility: float) -> float:
        """Berechnet die Konfidenz der Vorhersage"""
        
        # Basis-Konfidenz basierend auf der Vorhersage-Stärke
        base_confidence = min(abs(pred_change) * 10, 100)
        
        # Reduziere Konfidenz bei hoher Volatilität
        volatility_factor = max(0, 1 - (volatility - 20) / 100)
        
        return base_confidence * volatility_factor
    
    def _generate_trading_recommendation(self, price_change: float, pred_change: float, 
                                      rsi: float, trend: str) -> str:
        """Generiert eine Handelsempfehlung"""
        
        signals = []
        
        # Trend-basierte Signale
        if trend == "aufwärts":
            if pred_change > 2:
                signals.append("Starker Aufwärtstrend mit positiver Vorhersage")
            elif pred_change < -2:
                signals.append("Aufwärtstrend könnte sich umkehren")
        else:
            if pred_change < -2:
                signals.append("Starker Abwärtstrend mit negativer Vorhersage")
            elif pred_change > 2:
                signals.append("Abwärtstrend könnte sich umkehren")
                
        # RSI-basierte Signale
        if rsi > 70:
            signals.append("Überkaufte Bedingungen - Vorsicht bei Käufen")
        elif rsi < 30:
            signals.append("Überverkaufte Bedingungen - Kaufgelegenheit möglich")
            
        # Momentum-basierte Signale
        if price_change > 2 and pred_change > 2:
            signals.append("Starkes positives Momentum")
        elif price_change < -2 and pred_change < -2:
            signals.append("Starkes negatives Momentum")
            
        # Generiere Empfehlung
        if not signals:
            return "   Keine klaren Signale - neutrales Marktumfeld"
            
        recommendation = "   Basierend auf der Analyse:\n"
        for signal in signals:
            recommendation += f"   - {signal}\n"
            
        return recommendation
    
    def _assess_risk(self, volatility: float, volume_change: float, trend_strength: float) -> str:
        """Bewertet das Risiko"""
        
        risks = []
        
        # Volatilitäts-Risiko
        if volatility > 30:
            risks.append(f"HOCH: Sehr hohe Volatilität ({volatility:.1f}%)")
        elif volatility > 20:
            risks.append(f"MITTEL: Erhöhte Volatilität ({volatility:.1f}%)")
        else:
            risks.append(f"NIEDRIG: Moderate Volatilität ({volatility:.1f}%)")
            
        # Volumen-Risiko
        if abs(volume_change) > 50:
            risks.append(f"HOCH: Extreme Volumenänderung ({volume_change:+.1f}%)")
        elif abs(volume_change) > 20:
            risks.append(f"MITTEL: Deutliche Volumenänderung ({volume_change:+.1f}%)")
            
        # Trend-Risiko
        if trend_strength > 10:
            risks.append(f"MITTEL: Starker Trend ({trend_strength:.1f}%) - Trendumkehr möglich")
            
        # Generiere Risikobewertung
        if not risks:
            return "   Keine besonderen Risikofaktoren identifiziert"
            
        assessment = "   Identifizierte Risikofaktoren:\n"
        for risk in risks:
            assessment += f"   - {risk}\n"
            
        return assessment 