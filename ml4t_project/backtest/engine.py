"""
Backtesting Engine für ML4T
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class BacktestEngine:
    """Engine für das Backtesting von Trading-Strategien"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.001
    ):
        """
        Initialisiert die Backtest-Engine.
        
        Args:
            initial_capital: Startkapital
            commission: Kommission pro Trade (%)
            slippage: Slippage pro Trade (%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Setzt die Engine auf Anfangszustand zurück"""
        self.capital = self.initial_capital
        self.position = 0
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_capital]
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Führt den Backtest durch.
        
        Args:
            data: DataFrame mit OHLCV-Daten und Signalen
            
        Returns:
            Dict mit Backtest-Ergebnissen
        """
        self.reset()
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['Close']
            
            # Simuliere Trading mit Slippage
            executed_price = current_price * (1 + self.slippage)
            
            # Berechne Positionswert
            position_value = self.position * current_price
            
            # Aktualisiere Equity
            self.equity_curve.append(self.capital + position_value)
        
        return self._calculate_statistics()
    
    def _calculate_statistics(self) -> Dict:
        """Berechnet Performance-Statistiken"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        stats = {
            'total_return': (equity_series.iloc[-1] / self.initial_capital - 1) * 100,
            'max_drawdown': self._calculate_max_drawdown(equity_series) * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'num_trades': len(self.trades),
            'final_capital': self.equity_curve[-1]
        }
        
        return stats
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Berechnet den maximalen Drawdown"""
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Berechnet die Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        
        # Annualisierte Sharpe Ratio (252 Handelstage)
        return np.sqrt(252) * returns.mean() / returns.std() 