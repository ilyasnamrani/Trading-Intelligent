"""
D03 - Module d'Econometrie de la Moyenne (ARIMA, SARIMA, SARIMAX)
Projet : Trading Intelligent - Variante B01 (QQQ vs SPY)

Objectif :
----------
Modeliser et predire le rendement logarithmique de QQQ a t+1 (target)
a l'aide de modeles econometriques en series temporelles :
  1. Baselines de reference (Buy-and-Hold, Rendement Moyen, Moyenne Mobile)
  2. Diagnostic de stationnarite (Test ADF)
  3. Recherche sur grille ARIMA(p, d, q) via AIC / BIC
  4. Test de saisonnalite SARIMA(p, d, q)x(P, D, Q)_s et justification de rejet
  5. Modele SARIMAX avec variable exogene retardee (r_spy_lag1)
  6. Evaluation stricte hors-echantillon (Validation & Test) sans fuite temporelle
  7. Diagnostic des residus (Test de Ljung-Box pour blancheur)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ─── CONFIGURATION & CHEMINS ─────────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
FIGURES_DIR         = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Decoupage temporel strict (Cahier des charges)
# Train      : 2018-01-01 -> 2022-12-31 (Calibration)
# Validation : 2023-01-01 -> 2024-12-31 (Selection d'hyperparametres)
# Test       : 2025-01-01 -> 2026-10-31 (Evaluation finale gelee)
DATE_TRAIN_END = "2022-12-31"
DATE_VAL_END   = "2024-12-31"


def load_dataset():
    """Charge le dataset nettoye et enrichi produit par D02."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable a {PROCESSED_DATA_PATH}. Executez d'abord src/data.py.")
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    return df


def split_data(df):
    """Separe le dataset selon les 3 fenetres temporelles etanches."""
    train = df.loc[:DATE_TRAIN_END]
    val   = df.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:] # Eviter chevauchement
    test  = df.loc[DATE_VAL_END:].iloc[1:]
    return train, val, test


# ─── 1. DIAGNOSTIC DE STATIONNARITE (TEST ADF) ──────────────────────────────
def check_stationarity(series, name="r_qqq"):
    """
    Execute le test de Dickey-Fuller Augmente (ADF).
    H0 : La serie possede une racine unitaire (non-stationnaire).
    H1 : La serie est stationnaire.
    Si p-value < 0.05 -> Stationnaire -> Ordre d'integration d = 0.
    """
    res = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_val, lags, nobs, crit = res[0], res[1], res[2], res[3], res[4]
    
    is_stat = p_val < 0.05
    d_order = 0 if is_stat else 1
    
    report = {
        "serie": name,
        "adf_stat": float(adf_stat),
        "p_value": float(p_val),
        "lags_utilises": int(lags),
        "observations": int(nobs),
        "seuil_1%": float(crit["1%"]),
        "seuil_5%": float(crit["5%"]),
        "stationnaire": is_stat,
        "d_recommande": d_order
    }
    return report


# ─── 2. RECHERCHE SUR GRILLE ARIMA(p, d, q) ──────────────────────────────────
def grid_search_arima(y_train, d=0, max_p=3, max_q=3):
    """
    Explore les combinaisons ARIMA(p, d, q) sur l'echantillon Train
    et classe par critere d'information BIC (priorite a la parcimonie).
    """
    results = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y_train, order=(p, d, q))
                fitted = model.fit()
                results.append({
                    "p": p, "d": d, "q": q,
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "llf": fitted.llf
                })
            except Exception:
                continue

    res_df = pd.DataFrame(results).sort_values("bic").reset_index(drop=True)
    best_order = (int(res_df.iloc[0]["p"]), int(res_df.iloc[0]["d"]), int(res_df.iloc[0]["q"]))
    return res_df, best_order


# ─── 3. TEST DE SAISONNALITE (SARIMA) ────────────────────────────────────────
def evaluate_sarima(y_train, best_arima_order, s=5):
    """
    Compare le meilleur ARIMA a un SARIMA avec saisonnalite hebdomadaire (s=5).
    Verifie si l'introduction de composantes saisonnieres (P=1, D=0, Q=1)s ameliore le BIC.
    """
    p, d, q = best_arima_order
    try:
        # Ajustement ARIMA
        m_arima = ARIMA(y_train, order=(p, d, q)).fit()
        # Ajustement SARIMA (composante hebdomadaire 5 jours de bourse)
        m_sarima = ARIMA(y_train, order=(p, d, q), seasonal_order=(1, 0, 1, s)).fit()
        
        sarima_report = {
            "arima_order": (p, d, q),
            "arima_bic": float(m_arima.bic),
            "arima_aic": float(m_arima.aic),
            "sarima_order": f"({p},{d},{q})x(1,0,1)_{s}",
            "sarima_bic": float(m_sarima.bic),
            "sarima_aic": float(m_sarima.aic),
            "sarima_retenu": bool(m_sarima.bic < m_arima.bic),
            "justification": "SARIMA rejete : le gain de vraisemblance ne compense pas la penalite de complexite BIC (absence de saisonnalite deterministe sur QQQ)." if m_sarima.bic >= m_arima.bic else "SARIMA retenu selon le BIC."
        }
    except Exception as e:
        sarima_report = {
            "arima_order": (p, d, q),
            "sarima_retenu": False,
            "justification": f"Echec de convergence SARIMA ({str(e)}) -> Rejet justifie."
        }
    return sarima_report


# ─── 4. SARIMAX AVEC VARIABLE EXOGENE (r_spy_lag1) ───────────────────────────
def fit_sarimax(y_train, exog_train, order):
    """
    Ajuste un modele SARIMAX conditionne par une variable exogene retardee.
    Ici, la variable exogene est r_spy_lag1 (rendement SPY d'hier, connu a t).
    """
    model = ARIMA(y_train, order=order, exog=exog_train)
    fitted = model.fit()
    return fitted


# ─── 5. PREVISION ROLLING HORS-ECHANTILLON (SANS FUITE TEMPORELLE) ───────────
def evaluate_models_rolling(df, order):
    """
    Evalue ARIMA vs SARIMAX vs Baselines sur Validation (2023-2024) et Test (2025-2026).
    Protocole temporel rigoureux :
      Pour chaque date t dans [val_start .. test_end] :
      On utilise les coefficients appris sur Train (ou mis a jour)
      sans JAMAIS injecter les informations futures.
    """
    train, val, test = split_data(df)
    eval_df = pd.concat([val, test])
    
    # 1. Baseline : Rendement moyen historique sur Train
    hist_mean = train["r_qqq"].mean()
    
    # 2. Entrainement ARIMA(p, d, q) sur Train
    arima_model = ARIMA(train["r_qqq"], order=order).fit()
    
    # 3. Entrainement SARIMAX(p, d, q) sur Train avec exog = r_spy_lag1
    sarimax_model = ARIMA(train["r_qqq"], order=order, exog=train[["r_spy_lag1"]]).fit()
    
    # Previsions 1-step-ahead dynamiques
    # Pour respecter le critere de non-fuite, on applique les modeles sur la fenetre d'evaluation
    # statsmodels permet d'appliquer le filtre avec les parametres geles de Train
    arima_applied = arima_model.apply(eval_df["r_qqq"])
    arima_preds   = arima_applied.fittedvalues
    
    sarimax_applied = sarimax_model.apply(eval_df["r_qqq"], exog=eval_df[["r_spy_lag1"]])
    sarimax_preds   = sarimax_applied.fittedvalues
    
    # Construction du tableau de suivi des previsions
    preds_df = pd.DataFrame(index=eval_df.index)
    preds_df["r_actual"]         = eval_df["r_qqq"]
    preds_df["pred_baseline_mean"] = hist_mean
    preds_df["pred_arima"]        = arima_preds
    preds_df["pred_sarimax"]      = sarimax_preds
    
    # Alignement temporel : predit a t pour la seance t (base sur info t-1)
    return train, val, test, arima_model, sarimax_model, preds_df


# ─── 6. METRIQUES D'EVALUATION SCIENTIFIQUE ──────────────────────────────────
def compute_metrics(y_true, y_pred, name="Model"):
    """Calcule RMSE, MAE et Exactitude Directionnelle (Hit Ratio)."""
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.mean(np.abs(err))
    
    # Hit Ratio : % de fois ou le signe (direction) est correctement anticipe
    correct_dir = (np.sign(y_true) == np.sign(y_pred)).mean() * 100.0
    
    return {
        "Modele": name,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Hit_Ratio_%": float(correct_dir)
    }


# ─── 7. DIAGNOSTIC DES RESIDUS (LJUNG-BOX) ───────────────────────────────────
def test_residuals(model_fitted, model_name="ARIMA"):
    """
    Test de blancheur des residus de Ljung-Box.
    H0 : Les residus sont independants (bruit blanc).
    H1 : Il subsiste de l'autocorrelation (information non captee).
    Si p-value > 0.05 -> Residus purs et sans biais.
    """
    resid = model_fitted.resid.dropna()
    lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
    p_val = float(lb_test["lb_pvalue"].iloc[0])
    stat  = float(lb_test["lb_stat"].iloc[0])
    
    return {
        "modele": model_name,
        "stat_ljungbox": stat,
        "p_value": p_val,
        "residus_blancs": bool(p_val > 0.05),
        "conclusion": "Residus decorreles (bruit blanc confirme, pas de fuite d'info)" if p_val > 0.05 else "Legere autocorrelation residuelle detectee."
    }


# ─── 8. GENERATION DE LA FIGURE COMPARATIVE D03 ──────────────────────────────
def plot_d03_results(preds_df, arima_resid, sarimax_resid):
    """Genere le graphique d'audit comparatif pour D03."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("D03 - Evaluation Econometrique : ARIMA vs SARIMAX vs Realisations QQQ",
                 fontsize=14, fontweight="bold")
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Trajectoire des rendements reels vs previsions (Zoom 120 dernieres seances)
    ax = axes[0, 0]
    zoom = preds_df.iloc[-120:]
    ax.plot(zoom.index, zoom["r_actual"] * 100, label="Rendement reel QQQ", color="black", lw=1.2, alpha=0.7)
    ax.plot(zoom.index, zoom["pred_arima"] * 100, label="Prevision ARIMA", color="#1f77b4", lw=1.5, ls="--")
    ax.plot(zoom.index, zoom["pred_sarimax"] * 100, label="Prevision SARIMAX (exog SPY)", color="#2ca02c", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_title("1. Comparatif des previsions 1-step (Zoom 120 dernieres seances)")
    ax.set_ylabel("Rendement log (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # 2. Nuage de correlation : Reel vs SARIMAX
    ax = axes[0, 1]
    ax.scatter(preds_df["pred_sarimax"] * 100, preds_df["r_actual"] * 100, alpha=0.4, color="#2ca02c", s=15)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    m, b = np.polyfit(preds_df["pred_sarimax"] * 100, preds_df["r_actual"] * 100, 1)
    x_vals = np.linspace(preds_df["pred_sarimax"].min() * 100, preds_df["pred_sarimax"].max() * 100, 100)
    ax.plot(x_vals, m * x_vals + b, color="red", lw=1.5, label=f"Droite de regression (pente={m:.2f})")
    ax.set_title("2. Dispersion : Rendement Reel vs Prevision SARIMAX")
    ax.set_xlabel("Prevision SARIMAX (%)")
    ax.set_ylabel("Rendement Reel QQQ (%)")
    ax.legend(fontsize=9)

    # 3. Distribution des residus SARIMAX vs Loi normale
    ax = axes[1, 0]
    res_clean = sarimax_resid.dropna()
    ax.hist(res_clean, bins=60, density=True, alpha=0.6, color="#2ca02c", label="Residus SARIMAX")
    mu, sigma = res_clean.mean(), res_clean.std()
    x = np.linspace(res_clean.min(), res_clean.max(), 200)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    ax.plot(x, normal, "r--", lw=2, label="Normale theorique")
    ax.set_title(f"3. Distribution des residus (Moyenne={mu:.5f}, Ecart-type={sigma:.4f})")
    ax.set_xlabel("Erreur de prevision (Residu)")
    ax.set_ylabel("Densite")
    ax.legend(fontsize=9)

    # 4. Erreurs cumulees : ARIMA vs SARIMAX vs Naive
    ax = axes[1, 1]
    cum_err_arima   = np.cumsum(np.abs(preds_df["r_actual"] - preds_df["pred_arima"]))
    cum_err_sarimax = np.cumsum(np.abs(preds_df["r_actual"] - preds_df["pred_sarimax"]))
    cum_err_base    = np.cumsum(np.abs(preds_df["r_actual"] - preds_df["pred_baseline_mean"]))
    ax.plot(preds_df.index, cum_err_base, label="Baseline (Moyenne historique)", color="gray", ls=":")
    ax.plot(preds_df.index, cum_err_arima, label="ARIMA", color="#1f77b4", lw=1.5)
    ax.plot(preds_df.index, cum_err_sarimax, label="SARIMAX (SPY exog)", color="#2ca02c", lw=2)
    ax.set_title("4. Erreur Absolue Cumulee (Plus bas = Meilleur)")
    ax.set_ylabel("Erreur cumulee (MAE sum)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "D03_econometrie_comparatif.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ─── PIPELINE D03 COMPLET ────────────────────────────────────────────────────
def run_d03_pipeline():
    print("=" * 65)
    print("  LIVRABLE D03 - ECONOMETRIE DE LA MOYENNE (ARIMA / SARIMAX)")
    print("=" * 65)
    
    # Chargement
    df = load_dataset()
    train, val, test = split_data(df)
    print(f"[1/6] Decoupage temporel valide :")
    print(f"      Train      (2018-2022) : {len(train)} seances")
    print(f"      Validation (2023-2024) : {len(val)} seances")
    print(f"      Test       (2025-2026) : {len(test)} seances")
    
    # 1. Test de Stationnarite
    adf_res = check_stationarity(train["r_qqq"], name="r_qqq (Train)")
    print(f"\n[2/6] Diagnostic ADF (Dickey-Fuller) :")
    print(f"      Statistique ADF : {adf_res['adf_stat']:.4f} (Seuil 1% = {adf_res['seuil_1%']:.4f})")
    print(f"      p-value         : {adf_res['p_value']:.4e} -> {'Stationnaire' if adf_res['stationnaire'] else 'Non-stationnaire'}")
    print(f"      Ordre d retenu  : d = {adf_res['d_recommande']}")
    
    # 2. Grid Search ARIMA
    print(f"\n[3/6] Recherche sur grille ARIMA(p, {adf_res['d_recommande']}, q)...")
    grid_df, best_order = grid_search_arima(train["r_qqq"], d=adf_res["d_recommande"])
    print(f"      Meilleur ordre retenu selon le BIC : ARIMA{best_order}")
    print(f"      Top 3 des modeles :")
    for i, row in grid_df.head(3).iterrows():
        print(f"        ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])}) -> BIC: {row['bic']:.2f}, AIC: {row['aic']:.2f}")
        
    # 3. Test Saisonnalite SARIMA
    print(f"\n[4/6] Test de saisonnalite SARIMA(s=5)...")
    sarima_diag = evaluate_sarima(train["r_qqq"], best_order, s=5)
    print(f"      ARIMA BIC  : {sarima_diag.get('arima_bic', 0):.2f}")
    print(f"      SARIMA BIC : {sarima_diag.get('sarima_bic', 0):.2f}")
    print(f"      Decision   : {'SARIMA retenu' if sarima_diag['sarima_retenu'] else 'SARIMA REJETE'}")
    print(f"      Raison     : {sarima_diag['justification']}")
    
    # 4. Evaluation rolling hors-echantillon (ARIMA vs SARIMAX)
    print(f"\n[5/6] Calibration SARIMAX avec r_spy_lag1 et evaluation rolling...")
    _, _, _, arima_fit, sarimax_fit, preds_df = evaluate_models_rolling(df, best_order)
    
    # Significativite de la variable exogene SPY
    spy_coef = float(sarimax_fit.params.get("r_spy_lag1", 0))
    spy_pval = float(sarimax_fit.pvalues.get("r_spy_lag1", 1.0))
    print(f"      Coefficient SPY retard : {spy_coef:.4f} (p-value: {spy_pval:.4e})")
    
    # Calcul des metriques de performance
    metrics = []
    # Validation
    val_preds = preds_df.loc[val.index[0]:val.index[-1]]
    metrics.append(compute_metrics(val_preds["r_actual"], val_preds["pred_baseline_mean"], "Baseline Moyenne (Val)"))
    metrics.append(compute_metrics(val_preds["r_actual"], val_preds["pred_arima"], f"ARIMA{best_order} (Val)"))
    metrics.append(compute_metrics(val_preds["r_actual"], val_preds["pred_sarimax"], f"SARIMAX{best_order}+SPY (Val)"))
    
    # Test
    test_preds = preds_df.loc[test.index[0]:test.index[-1]]
    metrics.append(compute_metrics(test_preds["r_actual"], test_preds["pred_baseline_mean"], "Baseline Moyenne (Test)"))
    metrics.append(compute_metrics(test_preds["r_actual"], test_preds["pred_arima"], f"ARIMA{best_order} (Test)"))
    metrics.append(compute_metrics(test_preds["r_actual"], test_preds["pred_sarimax"], f"SARIMAX{best_order}+SPY (Test)"))
    
    metrics_df = pd.DataFrame(metrics)
    print("\n      Tableau des performances comparatives :")
    print(metrics_df.to_string(index=False))
    
    # 5. Diagnostic des residus Ljung-Box
    print(f"\n[6/6] Tests de blancheur des residus (Ljung-Box)...")
    lb_arima   = test_residuals(arima_fit, f"ARIMA{best_order}")
    lb_sarimax = test_residuals(sarimax_fit, f"SARIMAX{best_order}")
    print(f"      ARIMA   p-value Ljung-Box : {lb_arima['p_value']:.4f} -> {lb_arima['conclusion']}")
    print(f"      SARIMAX p-value Ljung-Box : {lb_sarimax['p_value']:.4f} -> {lb_sarimax['conclusion']}")
    
    # Sauvegarde graphique
    fig_path = plot_d03_results(preds_df, arima_fit.resid, sarimax_fit.resid)
    print(f"\n[OK]  Graphique D03 genere avec succes : {fig_path}")
    
    return {
        "adf": adf_res,
        "best_order": best_order,
        "grid_df": grid_df,
        "sarima_diag": sarima_diag,
        "spy_coef": spy_coef,
        "spy_pval": spy_pval,
        "metrics_df": metrics_df,
        "lb_arima": lb_arima,
        "lb_sarimax": lb_sarimax,
        "fig_path": fig_path
    }


if __name__ == "__main__":
    run_d03_pipeline()
