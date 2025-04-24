from typing import List, Dict

class Portfolio:
    """Verwaltet ein Portfolio mit Positionen und Performance-Tracking."""
    
    def __init__(self, initial_capital: float = 10_000.0):
        """
        Initialisiert ein neues Portfolio.
        
        Args:
            initial_capital: Startkapital (Standard: 10.000)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.position_value = 0.0
        self.total_value = initial_capital
        self.history = []
        
    def has_position(self) -> bool:
        """Prüft, ob eine offene Position existiert."""
        return self.position > 0
        
    def open_position(self, price: float, date: str) -> None:
        """
        Öffnet eine neue Position.
        
        Args:
            price: Aktueller Preis
            date: Handelsdatum
        """
        if not self.has_position():
            shares = self.cash / price
            self.position = shares
            self.cash = 0
            self.position_value = shares * price
            self._record_trade(date, 'BUY', shares, price)
            
    def close_position(self, price: float, date: str) -> None:
        """
        Schließt die aktuelle Position.
        
        Args:
            price: Aktueller Preis
            date: Handelsdatum
        """
        if self.has_position():
            self.cash = self.position * price
            shares = self.position
            self.position = 0
            self.position_value = 0
            self._record_trade(date, 'SELL', shares, price)
            
    def update_value(self, price: float, date: str) -> None:
        """
        Aktualisiert den Portfolio-Wert.
        
        Args:
            price: Aktueller Preis
            date: Handelsdatum
        """
        self.position_value = self.position * price
        self.total_value = self.cash + self.position_value
        self._record_value(date, price)
        
    def _record_trade(self, date: str, action: str, shares: float, price: float) -> None:
        """
        Zeichnet einen Handel auf.
        
        Args:
            date: Handelsdatum
            action: Handelsrichtung ('BUY' oder 'SELL')
            shares: Anzahl der Aktien
            price: Handelspreis
        """
        self.history.append({
            'date': date,
            'action': action,
            'shares': shares,
            'price': price,
            'value': self.total_value
        })
        
    def _record_value(self, date: str, price: float) -> None:
        """
        Zeichnet den Portfolio-Wert auf.
        
        Args:
            date: Datum
            price: Aktueller Preis
        """
        self.history.append({
            'date': date,
            'action': 'HOLD',
            'shares': self.position,
            'price': price,
            'value': self.total_value
        })
        
    def get_history(self) -> List[Dict]:
        """
        Gibt die Handelshistorie zurück.
        
        Returns:
            Liste von Handelsereignissen
        """
        return self.history 