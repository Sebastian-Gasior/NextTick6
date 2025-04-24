"""
Modul für Backtesting und Performance-Analyse
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from ml4t_project.signals.logic import Signal


@dataclass
class Position:
    """Repräsentiert eine offene Position"""
    entry_price: float
    size: int
    entry_date: pd.Timestamp


class Portfolio:
    """Verwaltet das Portfolio und führt Backtesting durch"""
    
    def __init__(self, initial_capital: float = 10_000.0):
        """
        Initialisiert das Portfolio
        
        Args:
            initial_capital: Startkapital
        """
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Setzt das Portfolio auf Anfangszustand zurück"""
        self.cash = self.initial_capital
        self.positions: List[Position] = []
        self.history: List[Dict] = []
    
    def execute_trade(self, signal: Signal, price: float, date: pd.Timestamp) -> None:
        """
        Führt einen Trade basierend auf Signal aus
        
        Args:
            signal: Handelssignal
            price: Aktueller Preis
            date: Aktuelles Datum
        """
        if signal == Signal.BUY and self.cash >= price:
            # Berechne maximale Anzahl Aktien die gekauft werden können
            max_shares = int(self.cash / price)
            if max_shares > 0:
                self.positions.append(Position(price, max_shares, date))
                self.cash -= max_shares * price
                
        elif signal == Signal.SELL and self.positions:
            # Verkaufe alle offenen Positionen
            for position in self.positions:
                self.cash += position.size * price
            self.positions = []
    
    def update_value(self, current_price: float, date: pd.Timestamp) -> float:
        """
        Aktualisiert und gibt den aktuellen Portfolio-Wert zurück
        
        Args:
            current_price: Aktueller Preis
            date: Aktuelles Datum
            
        Returns:
            Aktueller Portfolio-Wert
        """
        positions_value = sum(pos.size * current_price for pos in self.positions)
        total_value = self.cash + positions_value
        
        self.history.append({
            "date": date,
            "cash": self.cash,
            "positions_value": positions_value,
            "total_value": total_value
        })
        
        return total_value


def run_backtest(prices: np.ndarray,
                signals: np.ndarray,
                initial_capital: float = 10000.0,
                position_size: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
    """
    Führt einen Backtest der Trading-Strategie durch.
    
    Args:
        prices: Array mit Preisdaten
        signals: Array mit Trading-Signalen (1=Kauf, -1=Verkauf, 0=Halten)
        initial_capital: Startkapital
        position_size: Maximale Positionsgröße (0-1)
        
    Returns:
        DataFrame mit Backtest-Ergebnissen und Metriken-Dictionary
    """
    # Initialisierung
    portfolio_value = np.zeros(len(prices))
    portfolio_value[0] = initial_capital
    position = 0
    cash = initial_capital
    
    # Backtest durchführen
    for i in range(1, len(prices)):
        # Position aktualisieren
        if signals[i] == 1 and position == 0:  # Kauf
            shares = (cash * position_size) / prices[i]
            position = shares
            cash -= shares * prices[i]
        elif signals[i] == -1 and position > 0:  # Verkauf
            cash += position * prices[i]
            position = 0
            
        # Portfolio-Wert berechnen
        portfolio_value[i] = cash + (position * prices[i])
    
    # Ergebnisse aufbereiten
    results = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'returns': np.diff(portfolio_value, prepend=initial_capital) / portfolio_value,
        'position': np.where(position > 0, 1, 0)
    })
    
    # Metriken berechnen
    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    max_drawdown = np.min(portfolio_value / np.maximum.accumulate(portfolio_value) - 1)
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_value[-1]
    }
    
    return results, metrics


def calculate_metrics(
    portfolio_values: List[float],
    benchmark_values: List[float]
) -> Dict:
    """
    Berechnet Performance-Metriken
    
    Args:
        portfolio_values: Portfolio-Werte
        benchmark_values: Benchmark-Werte
        
    Returns:
        Dictionary mit Metriken
    """
    # Berechne Returns
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    
    # Gesamtrendite
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Sharpe Ratio (angenommen risikofreier Zins = 0)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    
    # Maximum Drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values[1:]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': np.std(portfolio_returns) * np.sqrt(252),
        'benchmark_return': (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
    } 