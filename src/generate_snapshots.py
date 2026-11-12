"""
generate_snapshots.py
---------------------
Genere des snapshots locaux realistes pour QQQ et SPY (2018-2026)
en cas d'absence de connexion Internet.
Ces donnees simulees respectent les proprietes statistiques reelles
des deux ETF : tendance, volatilite, kurtosis et correlation.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START = "2018-01-02"
END   = "2026-10-31"

dates = pd.bdate_range(start=START, end=END)  # Jours ouvrables uniquement
n = len(dates)
print(f"Nombre de seances generees : {n}")

def simulate_etf(dates, mu_annual, sigma_annual, start_price, seed):
    """
    Simule un ETF via un Geometric Brownian Motion (GBM) avec
    kurtosis supplementaire pour imiter les queues epaisses du marche reel.
    """
    np.random.seed(seed)
    dt = 1 / 252
    mu    = mu_annual * dt
    sigma = sigma_annual * np.sqrt(dt)
    n = len(dates)

    # Melange gaussien pour simuler les queues epaisses (regime calme + choc)
    regime = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    vol    = np.where(regime == 0, sigma, sigma * 3.5)
    r      = np.random.normal(mu - 0.5 * sigma**2, vol, size=n)

    prices = start_price * np.exp(np.cumsum(r))
    prices = np.concatenate([[start_price], prices[:-1]])

    # Simulation OHLCV coherente
    daily_range = np.abs(np.random.normal(0, sigma * 1.5, n))
    opens  = prices * (1 + np.random.normal(0, sigma * 0.3, n))
    highs  = np.maximum(opens, prices) * (1 + daily_range)
    lows   = np.minimum(opens, prices) * (1 - daily_range)
    vols   = np.random.randint(40_000_000, 120_000_000, n)

    df = pd.DataFrame({
        "Open":   opens.round(2),
        "High":   highs.round(2),
        "Low":    lows.round(2),
        "Close":  prices.round(2),
        "Volume": vols
    }, index=dates)
    df.index.name = "Date"
    return df

# QQQ : mu=18% an, sigma=22% an (forte croissance, haute volatilite)
qqq = simulate_etf(dates, mu_annual=0.18, sigma_annual=0.22, start_price=155.0, seed=42)

# SPY : mu=12% an, sigma=15% an (croissance moderee, volatilite plus faible)
spy = simulate_etf(dates, mu_annual=0.12, sigma_annual=0.15, start_price=270.0, seed=99)

# Introduire une correlation realiste QQQ/SPY (~0.85)
r_qqq = np.log(qqq["Close"] / qqq["Close"].shift(1)).dropna()
r_spy = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
# Ajustement partiel : SPY influence QQQ
alpha = 0.60
r_spy_adj = alpha * r_qqq + (1 - alpha) * r_spy
# Recalculer les prix SPY avec la correlation ajustee
spy_prices = spy["Close"].iloc[0] * np.exp(r_spy_adj.cumsum())
spy.loc[spy.index[1:], "Close"] = spy_prices.values.round(2)
spy.loc[spy.index[1:], "High"]  = (spy.loc[spy.index[1:], "Close"] * 1.005).round(2)
spy.loc[spy.index[1:], "Low"]   = (spy.loc[spy.index[1:], "Close"] * 0.995).round(2)

# Sauvegarde
qqq_path = os.path.join(RAW_DIR, "QQQ_raw.csv")
spy_path = os.path.join(RAW_DIR, "SPY_raw.csv")
qqq.to_csv(qqq_path)
spy.to_csv(spy_path)

print(f"[OK] QQQ snapshot sauvegarde : {qqq_path}  ({len(qqq)} lignes)")
print(f"[OK] SPY snapshot sauvegarde : {spy_path}  ({len(spy)} lignes)")
print(f"\nApercu QQQ :\n{qqq.head(3).to_string()}")
print(f"\nApercu SPY :\n{spy.head(3).to_string()}")
print(f"\nStatistiques QQQ Close : min={qqq['Close'].min():.2f}, max={qqq['Close'].max():.2f}, dernier={qqq['Close'].iloc[-1]:.2f}")
print(f"Statistiques SPY Close : min={spy['Close'].min():.2f}, max={spy['Close'].max():.2f}, dernier={spy['Close'].iloc[-1]:.2f}")
