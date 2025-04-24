"""
Modul zur Berechnung technischer Indikatoren
"""
import pandas as pd
import ta
import numpy as np
from typing import Tuple, Optional


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt alle konfigurierten technischen Indikatoren hinzu
    
    Args:
        df: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit zusätzlichen Indikatoren
    """
    # Füge alle Indikatortypen hinzu
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    
    # Entferne NaN-Werte (entstehen durch Berechnung der Indikatoren)
    df.dropna(inplace=True)
    
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Trend-Indikatoren hinzu (MACD, EMA etc.)
    
    Args:
        df: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit Trend-Indikatoren
    """
    # Validiere Daten (benötigt mindestens 200 Datenpunkte für EMA-200)
    validate_data(df['Close'], 200)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # EMAs
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Momentum-Indikatoren hinzu (RSI etc.)
    
    Args:
        df: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit Momentum-Indikatoren
    """
    # Validiere Daten (benötigt mindestens 14 Datenpunkte für RSI)
    validate_data(df['Close'], 14)
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # ROC (Rate of Change)
    df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Volatilitäts-Indikatoren hinzu (Bollinger Bands etc.)
    
    Args:
        df: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit Volatilitäts-Indikatoren
    """
    # Validiere Daten (benötigt mindestens 20 Datenpunkte für Bollinger Bands)
    validate_data(df['Close'], 20)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_UPPER'] = bollinger.bollinger_hband()
    df['BB_MIDDLE'] = bollinger.bollinger_mavg()
    df['BB_LOWER'] = bollinger.bollinger_lband()
    
    # Average True Range
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], 
        df['Low'], 
        df['Close']
    ).average_true_range()
    
    return df


def validate_data(data: pd.Series, min_length: int) -> None:
    """Validiere die Eingabedaten"""
    if len(data) < min_length:
        raise ValueError(f"Datenlänge muss mindestens {min_length} sein")
    if data.isna().any():
        raise ValueError("Daten enthalten NaN-Werte")


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Berechne den Relative Strength Index (RSI)"""
    validate_data(data, period)
    
    # Berechne tägliche Änderungen
    delta = data.diff()
    
    # Separate positive und negative Änderungen
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Berechne durchschnittliche Gewinne und Verluste
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Berechne RS und RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Berechne den Moving Average Convergence Divergence (MACD)"""
    validate_data(data, slow_period)
    
    # Berechne EMAs
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    
    # Berechne MACD Line
    macd_line = exp1 - exp2
    
    # Setze die ersten slow_period-1 Werte auf NaN
    macd_line.iloc[:slow_period-1] = np.nan
    
    # Berechne Signal Line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Berechne MACD Histogram
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Berechnet die Bollinger Bands für eine Zeitreihe.
    
    Args:
        data: Zeitreihe der Schlusskurse
        period: Periode für den gleitenden Durchschnitt
        std_dev: Anzahl der Standardabweichungen für die Bänder
    
    Returns:
        Tuple aus oberem Band, mittlerem Band und unterem Band
    """
    validate_data(data, period)
    
    # Berechne den gleitenden Durchschnitt (mittleres Band)
    middle_band = data.rolling(window=period).mean()
    
    # Berechne die Standardabweichung
    rolling_std = data.rolling(window=period).std()
    
    # Berechne die Bänder
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Setze die ersten period-1 Werte auf NaN
    upper_band.iloc[:period-1] = np.nan
    middle_band.iloc[:period-1] = np.nan
    lower_band.iloc[:period-1] = np.nan
    
    return upper_band, middle_band, lower_band


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Berechne den Exponential Moving Average (EMA)"""
    validate_data(data, period)
    
    # Berechne EMA
    ema = data.ewm(span=period, adjust=False).mean()
    
    # Setze die ersten period-1 Werte auf NaN
    ema.iloc[:period-1] = np.nan
    
    return ema


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet alle technischen Indikatoren für den Datensatz.
    
    Args:
        data: DataFrame mit OHLCV-Daten
        
    Returns:
        DataFrame mit allen Indikatoren
    """
    df = data.copy()
    
    # Trend Indikatoren
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Momentum Indikatoren
    df['rsi'] = ta.momentum.rsi(df['Close'])
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    
    # Volatilität Indikatoren
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['Close'])
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volumen Indikatoren
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['vwap'] = calculate_vwap(df)
    
    # Entferne NaN-Werte
    df = df.bfill().ffill()
    
    return df


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Berechnet den Volume Weighted Average Price (VWAP).
    
    Args:
        data: DataFrame mit OHLCV-Daten
        
    Returns:
        VWAP als Series
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap 