"""
Modul zur Visualisierung von Preisen, Signalen und Performance
"""
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional
from ml4t_project.signals.logic import Signal


def plot_predictions(
    dates: pd.DatetimeIndex,
    actual_prices: List[float],
    predicted_prices: List[float],
    title: str = "Preis vs. Vorhersage"
) -> go.Figure:
    """
    Plottet tatsächliche und vorhergesagte Preise
    
    Args:
        dates: Zeitindex
        actual_prices: Tatsächliche Preise
        predicted_prices: Vorhergesagte Preise
        title: Plot-Titel
        
    Returns:
        Plotly Figure-Objekt
    """
    fig = go.Figure()
    
    # Tatsächliche Preise
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_prices,
        name="Tatsächlich",
        line=dict(color="blue")
    ))
    
    # Vorhergesagte Preise
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_prices,
        name="Vorhersage",
        line=dict(color="red", dash="dash")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Preis",
        hovermode="x unified"
    )
    
    return fig


def plot_signals(
    dates: pd.DatetimeIndex,
    prices: List[float],
    signals: List[Signal],
    title: str = "Handelssignale"
) -> go.Figure:
    """
    Plottet Preise mit Handelssignalen
    
    Args:
        dates: Zeitindex
        prices: Preiszeitreihe
        signals: Handelssignale
        title: Plot-Titel
        
    Returns:
        Plotly Figure-Objekt
    """
    fig = go.Figure()
    
    # Preislinie
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        name="Preis",
        line=dict(color="blue")
    ))
    
    # Kaufsignale
    buy_dates = [date for date, signal in zip(dates, signals) if signal == Signal.BUY]
    buy_prices = [price for price, signal in zip(prices, signals) if signal == Signal.BUY]
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=buy_prices,
        mode="markers",
        name="Kauf",
        marker=dict(color="green", size=10, symbol="triangle-up")
    ))
    
    # Verkaufssignale
    sell_dates = [date for date, signal in zip(dates, signals) if signal == Signal.SELL]
    sell_prices = [price for price, signal in zip(prices, signals) if signal == Signal.SELL]
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=sell_prices,
        mode="markers",
        name="Verkauf",
        marker=dict(color="red", size=10, symbol="triangle-down")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Preis",
        hovermode="x unified"
    )
    
    return fig


def plot_performance(
    dates: pd.DatetimeIndex,
    portfolio_value: List[float],
    benchmark: Optional[List[float]] = None,
    title: str = "Portfolio Performance"
) -> go.Figure:
    """
    Plottet Portfolio-Performance
    
    Args:
        dates: Zeitindex
        portfolio_value: Portfolio-Werte
        benchmark: Optional Benchmark-Werte
        title: Plot-Titel
        
    Returns:
        Plotly Figure-Objekt
    """
    fig = go.Figure()
    
    # Portfolio-Wert
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_value,
        name="Portfolio",
        line=dict(color="blue")
    ))
    
    # Benchmark (falls vorhanden)
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark,
            name="Benchmark",
            line=dict(color="gray", dash="dash")
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Wert",
        hovermode="x unified"
    )
    
    return fig 