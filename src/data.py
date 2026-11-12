"""
D02 - Module d'ingestion, audit et ingenierie des donnees de marche.
Projet : Trading Intelligent - Variante B01 (QQQ / Nasdaq-100)
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

TICKERS    = ["QQQ", "SPY"]
START_DATE = "2018-01-01"
END_DATE   = "2026-10-31"

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR   = os.path.join("reports", "figures")

for d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── ETAPE 1 : TELECHARGEMENT ET SNAPSHOT LOCAL ──────────────────────────────

def download_data(ticker: str, force_download: bool = False) -> pd.DataFrame:
    """
    Telecharge les donnees OHLCV ajustees pour un ticker donne.

    Pourquoi Adj Close et pas Close ?
    - Le prix brut est biaise par les splits et dividendes.
    - Adj Close corrige ces artefacts pour que les rendements soient reels.

    Pourquoi sauvegarder localement ?
    - Garantir la reproductibilite et le mode hors-ligne (requis par le cahier
      des charges pour la demo).
    """
    snapshot_path = os.path.join(RAW_DIR, f"{ticker}_raw.csv")

    if os.path.exists(snapshot_path) and not force_download:
        print(f"[INFO] {ticker} -> Chargement depuis snapshot local : {snapshot_path}")
        df = pd.read_csv(snapshot_path, index_col=0, parse_dates=True)
        return df

    print(f"[INFO] {ticker} -> Telechargement Yahoo Finance ({START_DATE} -> {END_DATE})...")
    raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"Aucune donnee recue pour '{ticker}'. Verifiez la connexion.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.to_csv(snapshot_path)
    print(f"[OK]   {ticker} -> Snapshot sauvegarde : {snapshot_path} ({len(raw)} seances)")
    return raw


# ─── ETAPE 2 : AUDIT QUALITE ─────────────────────────────────────────────────

def audit_quality(df: pd.DataFrame, ticker: str) -> dict:
    """
    Controle d'integrite complet sur un DataFrame OHLCV.

    Checks effectues :
    1. Valeurs manquantes (NaN)
    2. Doublons d'index (meme date deux fois)
    3. Coherence OHLC : Low <= Open, Close <= High
    4. Volumes nuls (seances suspectes)
    5. Rendements extremes > +-15% (anomalies)
    """
    print(f"\n{'='*55}")
    print(f"  AUDIT QUALITE -- {ticker}")
    print(f"{'='*55}")

    report = {}

    report["date_debut"] = str(df.index.min().date())
    report["date_fin"]   = str(df.index.max().date())
    report["nb_seances"] = len(df)
    print(f"  Periode          : {report['date_debut']} -> {report['date_fin']}")
    print(f"  Nb de seances    : {report['nb_seances']}")

    missing = df.isnull().sum()
    report["valeurs_manquantes"] = missing.to_dict()
    print(f"  Valeurs NaN      : {missing.sum()} au total")

    n_dup = df.index.duplicated().sum()
    report["doublons"] = int(n_dup)
    print(f"  Doublons d'index : {n_dup}")

    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        inc_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
        inc_low  = (df["Low"]  > df[["Open", "Close"]].min(axis=1)).sum()
        report["incoherence_ohlc"] = int(inc_high + inc_low)
        print(f"  Incoherences OHLC: {report['incoherence_ohlc']}")

    if "Volume" in df.columns:
        n_zero = (df["Volume"] == 0).sum()
        report["volumes_nuls"] = int(n_zero)
        print(f"  Volumes = 0      : {n_zero}")

    if "Close" in df.columns:
        rets = df["Close"].pct_change().dropna()
        n_ext = (rets.abs() > 0.15).sum()
        report["rendements_extremes"] = int(n_ext)
        print(f"  Rendements >+/-15%: {n_ext}")

    ok = (missing.sum() == 0 and n_dup == 0 and report.get("incoherence_ohlc", 0) == 0)
    report["statut"] = "OK - PROPRES" if ok else "ATTENTION - A VERIFIER"
    print(f"  Statut global    : {report['statut']}")
    return report


# ─── ETAPE 3 : NETTOYAGE ─────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Nettoyage minimal : tri chronologique, suppression des doublons,
    interpolation des NaN sur max 1 jour (forward-fill).
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.ffill(limit=1)
    print(f"[OK]   {ticker} -> Nettoyage OK. {len(df)} seances retenues.")
    return df


# ─── ETAPE 4 : INGENIERIE DES FEATURES ───────────────────────────────────────

def build_features(qqq: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame final utilise par tous les modeles.

    Variables creees :
    ------------------
    close_qqq  : Prix ajuste de QQQ
    close_spy  : Prix ajuste de SPY (benchmark)
    r_qqq      : Rendement log de QQQ a t  [ln(P_t / P_{t-1})]
    r_spy      : Rendement log de SPY a t
    r_qqq_lag1 : Rendement QQQ a t-1 (inertie court terme)
    r_qqq_lag5 : Rendement QQQ a t-5 (effet semaine)
    r_spy_lag1 : Rendement SPY a t-1 (variable exogene pour SARIMAX - B01)
    vol_20     : Volatilite realisee sur 20 seances (thermometre du risque)
    target     : Rendement QQQ a t+1 --> VARIABLE CIBLE (jamais en input!)

    Principe No-Lookahead :
    Toute variable a la date t est calculee uniquement avec des prix connus
    jusqu'a t inclus. La variable 'target' est uniquement pour l'evaluation.
    """
    df = pd.DataFrame(index=qqq.index)

    df["close_qqq"] = qqq["Close"]
    df["close_spy"] = spy["Close"].reindex(qqq.index, method="ffill")

    # Rendements logarithmiques
    df["r_qqq"] = np.log(df["close_qqq"] / df["close_qqq"].shift(1))
    df["r_spy"] = np.log(df["close_spy"] / df["close_spy"].shift(1))

    # Variables retardees (lookback)
    df["r_qqq_lag1"] = df["r_qqq"].shift(1)
    df["r_qqq_lag5"] = df["r_qqq"].shift(5)
    df["r_spy_lag1"] = df["r_spy"].shift(1)

    # Volatilite realisee sur 20 seances
    df["vol_20"] = df["r_qqq"].rolling(window=20).std()

    # Variable cible : rendement de DEMAIN
    df["target"] = df["r_qqq"].shift(-1)

    df = df.dropna()

    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(out_path)
    print(f"\n[OK]   Dataset enrichi sauvegarde : {out_path}")
    print(f"       {len(df)} seances x {len(df.columns)} variables")
    return df


# ─── ETAPE 5 : VISUALISATIONS ────────────────────────────────────────────────

def plot_all(df: pd.DataFrame):
    """
    Genere les 4 graphiques d'audit requis pour le D02.

    Fig 1 : Prix normalises base 100 (QQQ vs SPY)
      -- Montre qui surperforme sur la periode.
      -- BENCHMARK : SPY = reference du marche large.

    Fig 2 : Rendements journaliers QQQ
      -- Revele les episodes de crise et d'euphorie.

    Fig 3 : Volatilite realisee 20j (annualisee)
      -- Montre visuellement les regimes de volatilite.
      -- C'est ce que GARCH essaiera de prevoir.

    Fig 4 : Distribution des rendements vs loi normale
      -- Si queues epaisses (kurtosis eleve) : risque de krach
         sous-estime par les modeles supposant la normalite.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("D02 -- Audit des donnees : QQQ (Nasdaq-100) vs SPY (S&P 500)",
                 fontsize=14, fontweight="bold")

    # -- Fig 1 : Prix normalises --
    ax = axes[0, 0]
    base_qqq = df["close_qqq"].iloc[0]
    base_spy = df["close_spy"].iloc[0]
    ax.plot(df.index, df["close_qqq"] / base_qqq * 100, label="QQQ (Nasdaq-100)", color="#1f77b4", lw=1.5)
    ax.plot(df.index, df["close_spy"] / base_spy * 100, label="SPY (S&P 500 - Benchmark)", color="#ff7f0e", lw=1.5, ls="--")
    ax.axhline(100, color="gray", lw=0.5, ls=":")
    ax.set_title("1. Prix normalises (Base 100 = Jan 2018)")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # -- Fig 2 : Rendements journaliers --
    ax = axes[0, 1]
    colors = ["#d62728" if r < 0 else "#2ca02c" for r in df["r_qqq"]]
    ax.bar(df.index, df["r_qqq"] * 100, color=colors, width=1, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("2. Rendements journaliers de QQQ (%)")
    ax.set_ylabel("Rendement log (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # -- Fig 3 : Volatilite realisee --
    ax = axes[1, 0]
    vol_ann = df["vol_20"] * np.sqrt(252) * 100
    ax.fill_between(df.index, vol_ann, alpha=0.4, color="#9467bd")
    ax.plot(df.index, vol_ann, color="#9467bd", lw=1)
    ax.axhline(vol_ann.mean(), color="red", lw=1, ls="--",
               label=f"Moyenne : {vol_ann.mean():.1f}%")
    ax.set_title("3. Volatilite realisee QQQ (20j, annualisee)")
    ax.set_ylabel("Volatilite annualisee (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # -- Fig 4 : Distribution des rendements --
    ax = axes[1, 1]
    r = df["r_qqq"].dropna() * 100
    ax.hist(r, bins=80, density=True, alpha=0.6, color="#1f77b4", label="Rendements QQQ")
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 300)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, "r--", lw=2, label="Loi normale theorique")
    ax.axvline(0, color="black", lw=0.5)
    kurt = float(r.kurtosis())
    ax.set_title(f"4. Distribution des rendements (Kurtosis = {kurt:.2f})")
    ax.set_xlabel("Rendement journalier (%)")
    ax.set_ylabel("Densite")
    ax.legend(fontsize=9)
    note = "Queues epaisses -> risque sous-estime" if kurt > 1 else "Proche de la normale"
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "D02_audit_donnees.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK]   Graphique D02 sauvegarde : {out_path}")


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_pipeline(force_download: bool = False) -> pd.DataFrame:
    """
    Execute l'integralite du pipeline D02.
    Retourne le DataFrame enrichi pret pour la modelisation.
    """
    print("=" * 55)
    print("  PIPELINE D02 -- DONNEES, AUDIT ET FEATURES")
    print("=" * 55)

    qqq_raw = download_data("QQQ", force_download=force_download)
    spy_raw = download_data("SPY", force_download=force_download)

    audit_quality(qqq_raw, "QQQ")
    audit_quality(spy_raw, "SPY")

    qqq_clean = clean_data(qqq_raw, "QQQ")
    spy_clean = clean_data(spy_raw, "SPY")

    df_features = build_features(qqq_clean, spy_clean)

    plot_all(df_features)

    print("\n" + "=" * 55)
    print("  D02 TERMINE -- Dataset pret pour la modelisation.")
    print("=" * 55)
    return df_features


if __name__ == "__main__":
    df = run_pipeline()
    print("\nApercu (5 premieres lignes) :")
    print(df.head().to_string())
    print("\nApercu (5 dernieres lignes) :")
    print(df.tail().to_string())
    print("\nDescription statistique :")
    print(df.describe().round(6).to_string())
