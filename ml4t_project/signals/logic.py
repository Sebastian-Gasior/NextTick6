"""
Modul zur Generierung von Handelssignalen
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from enum import Enum


class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


def generate_signals(predictions: np.ndarray,
                    prices: np.ndarray,
                    buy_threshold: float = 0.02,
                    sell_threshold: float = -0.02) -> np.ndarray:
    """
    Generiert Handelssignale basierend auf Vorhersagen.
    
    Args:
        predictions: Vorhersagewerte
        prices: Tatsächliche Preise
        buy_threshold: Schwellenwert für Kaufsignal
        sell_threshold: Schwellenwert für Verkaufssignal
        
    Returns:
        np.ndarray: Array mit Handelssignalen (1=Kauf, -1=Verkauf, 0=Halten)
    """
    signals = np.zeros(len(predictions))
    
    # Berechne prozentuale Änderung
    returns = (predictions - prices) / prices
    
    # Generiere Signale
    signals[returns > buy_threshold] = 1  # Kauf
    signals[returns < sell_threshold] = -1  # Verkauf
    
    return signals


def apply_signal_rules(
    signals: List[Signal],
    prices: np.ndarray,
    position_size: float = 0.1
) -> pd.DataFrame:
    """
    Wendet Handelsregeln auf Signale an
    
    Args:
        signals: Liste von Handelssignalen
        prices: Preiszeitreihe
        position_size: Größe der Position (0-1)
        
    Returns:
        DataFrame mit Trades und Positionen
    """
    trades = []
    position = 0
    entry_price = 0
    
    for i, (signal, price) in enumerate(zip(signals, prices)):
        trade = {
            'timestamp': i,
            'price': price,
            'signal': signal.value,
            'position': position,
            'trade_type': 'HOLD',
            'pnl': 0.0
        }
        
        # Kaufsignal und keine Position
        if signal == Signal.BUY and position == 0:
            position = position_size
            entry_price = price
            trade['trade_type'] = 'BUY'
            
        # Verkaufssignal und Long-Position
        elif signal == Signal.SELL and position > 0:
            pnl = position * (price - entry_price)
            position = 0
            trade['trade_type'] = 'SELL'
            trade['pnl'] = pnl
            
        trades.append(trade)
    
    return pd.DataFrame(trades) 