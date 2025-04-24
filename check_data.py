import yfinance as yf
import pandas as pd

# Lade AAPL-Daten
symbol = "AAPL"
start_date = "2022-01-01"
end_date = "2024-01-01"

# Daten abrufen
data = yf.download(symbol, start=start_date, end=end_date)

# Zeige detaillierte Informationen
print("\nDatenübersicht:")
print("-" * 50)
print(f"Anzahl Datenpunkte: {len(data)}")
print(f"Zeitraum: {data.index[0]} bis {data.index[-1]}")
print("\nErste 5 Zeilen:")
print(data.head())
print("\nLetzte 5 Zeilen:")
print(data.tail())
print("\nDatentypen:")
print(data.dtypes)
print("\nFehlende Werte:")
print(data.isnull().sum())
print("\nWertebereiche:")
for column in data.columns:
    print(f"{column}:")
    print(f"  Min: {data[column].min():.2f}")
    print(f"  Max: {data[column].max():.2f}")
    print(f"  Mittelwert: {data[column].mean():.2f}") 