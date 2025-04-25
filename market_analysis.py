"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class MarktAnalyse:
    def __init__(self, symbol: str):
        """
        Initialisiert die Marktanalyse für ein bestimmtes Symbol.
        
        Args:
            symbol (str): Das zu analysierende Aktiensymbol
        """
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        
    def hole_fundamentaldaten(self) -> Dict[str, Any]:
        """
        Holt die wichtigsten Fundamentaldaten des Unternehmens.
        
        Returns:
            Dict[str, Any]: Fundamentaldaten
        """
        info = self.stock.info
        return {
            'Name': info.get('longName', 'N/A'),
            'Sektor': info.get('sector', 'N/A'),
            'Industrie': info.get('industry', 'N/A'),
            'Marktkapitalisierung': info.get('marketCap', 'N/A'),
            'KGV': info.get('trailingPE', 'N/A'),
            'Dividendenrendite': info.get('dividendYield', 'N/A'),
        }
    
    def technische_analyse(self, zeitraum: str = '1y') -> pd.DataFrame:
        """
        Führt eine technische Analyse durch.
        
        Args:
            zeitraum (str): Analysezeitraum (z.B. '1y' für ein Jahr)
            
        Returns:
            pd.DataFrame: DataFrame mit technischen Indikatoren
        """
        # Historische Daten abrufen
        hist = self.stock.history(period=zeitraum)
        
        # Technische Indikatoren berechnen
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = self._berechne_rsi(hist['Close'])
        hist['MACD'] = self._berechne_macd(hist['Close'])
        
        return hist
    
    def _berechne_rsi(self, preise: pd.Series, perioden: int = 14) -> pd.Series:
        """
        Berechnet den Relative Strength Index (RSI).
        
        Args:
            preise (pd.Series): Preisdaten
            perioden (int): RSI-Periode
            
        Returns:
            pd.Series: RSI-Werte
        """
        delta = preise.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=perioden).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=perioden).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _berechne_macd(self, preise: pd.Series) -> pd.Series:
        """
        Berechnet den MACD (Moving Average Convergence Divergence).
        
        Args:
            preise (pd.Series): Preisdaten
            
        Returns:
            pd.Series: MACD-Werte
        """
        exp1 = preise.ewm(span=12, adjust=False).mean()
        exp2 = preise.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    def erstelle_bericht(self) -> str:
        """
        Erstellt einen ausführlichen Analysebericht.
        
        Returns:
            str: Formatierter Analysebericht
        """
        fundamental = self.hole_fundamentaldaten()
        technisch = self.technische_analyse()
        
        bericht = f"""
Marktanalyse für {self.symbol}
{'='*50}

Fundamentaldaten:
{'-'*20}
"""
        for key, value in fundamental.items():
            bericht += f"{key}: {value}\n"
            
        bericht += f"""
Technische Analyse:
{'-'*20}
Aktueller Preis: {technisch['Close'].iloc[-1]:.2f}
20-Tage SMA: {technisch['SMA_20'].iloc[-1]:.2f}
50-Tage SMA: {technisch['SMA_50'].iloc[-1]:.2f}
RSI: {technisch['RSI'].iloc[-1]:.2f}
MACD: {technisch['MACD'].iloc[-1]:.2f}

"""
        return bericht
    
    def visualisiere(self, zeitraum: str = '1y'):
        """
        Erstellt Visualisierungen der Analyse.
        
        Args:
            zeitraum (str): Analysezeitraum
        """
        daten = self.technische_analyse(zeitraum)
        
        # Plot erstellen
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Preis und SMAs
        ax1.plot(daten.index, daten['Close'], label='Schlusskurs')
        ax1.plot(daten.index, daten['SMA_20'], label='20-Tage SMA')
        ax1.plot(daten.index, daten['SMA_50'], label='50-Tage SMA')
        ax1.set_title(f'{self.symbol} Kursverlauf und SMAs')
        ax1.legend()
        
        # RSI
        ax2.plot(daten.index, daten['RSI'])
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI')
        
        # MACD
        ax3.plot(daten.index, daten['MACD'])
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('MACD')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Beispielnutzung
    analyse = MarktAnalyse("AAPL")
    print(analyse.erstelle_bericht())
    analyse.visualisiere()
""" 