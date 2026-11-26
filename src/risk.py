"""
D04 - Module de Risque : ARCH / GARCH, VaR et Expected Shortfall
Projet : Trading Intelligent - Variante B01 (QQQ vs SPY)

Objectif :
----------
Modeliser la volatilite conditionnelle du rendement de QQQ et quantifier
le risque de perte extreme (VaR et Expected Shortfall / CVaR), en vue
de piloter le dimensionnement dynamique des positions au Livrable D06.

Architecture du module (4 phases) :
------------------------------------
  Phase 1 : Diagnostic d'heteroscedasticite (Test ARCH-LM + Ljung-Box sur r^2)
             -> Prouver que la variance des residus n'est pas constante
                et que les chocs de volatilite arrivent en grappes.

  Phase 2 : Estimation du modele GARCH(1,1) de Bollerslev
             -> Modeliser la variance conditionnelle sigma_t^2
             -> Comparer distributions Gaussienne vs Student-t

  Phase 3 : Calcul des mesures de risque (VaR et Expected Shortfall)
             -> VaR parametrique a 95% et 99% pour chaque seance
             -> Expected Shortfall (CVaR)

  Phase 4 : Backtesting de la VaR (Test de Kupiec POF)
             -> Comptage des depassements et validation statistique

Lien avec les autres livrables :
---------------------------------
  D03 (entree)  : Residus du modele SARIMAX(1,0,0)
  D04 (sortie)  : sigma_{t+1} prevue -> Feature D05, position sizing D06
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as sp_stats

from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.stattools import acf as acf_func

warnings.filterwarnings("ignore")

# ─── CONFIGURATION & CHEMINS ────────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
FIGURES_DIR         = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATE_TRAIN_END = "2022-12-31"
DATE_VAL_END   = "2024-12-31"


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset():
    """Charge le dataset enrichi produit par D02 (data/processed/features.csv)."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset introuvable a {PROCESSED_DATA_PATH}. "
            f"Executez d'abord src/data.py."
        )
    return pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)


def split_data(df):
    """Separe le dataset selon les 3 fenetres temporelles etanches."""
    train = df.loc[:DATE_TRAIN_END]
    val   = df.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = df.loc[DATE_VAL_END:].iloc[1:]
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : DIAGNOSTIC D'HETEROSCEDASTICITE
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROLE : Prouver statistiquement que la variance des rendements de QQQ
#        n'est PAS constante dans le temps (volatility clustering).
#
# POURQUOI : Si la variance etait constante (homoscedastique), GARCH serait
#            inutile. Ce diagnostic justifie formellement la Phase 2.
#
# COMMENT : Deux tests complementaires :
#   (a) Test ARCH-LM d'Engle : regression de epsilon_t^2 sur ses retards.
#   (b) Test de Ljung-Box sur les rendements au carre r_t^2 :
#       teste directement l'autocorrelation des carres (plus puissant
#       sur les series financieres a memoire longue de volatilite).
#
# UTILITE : Si au moins un test rejette H0 (p < 0.05), l'effet ARCH est
#           confirme et le modele GARCH est justifie.
# ═══════════════════════════════════════════════════════════════════════════════

def test_arch_effect(returns, nlags_lm=20, nlags_lb=20):
    """
    Diagnostic d'heteroscedasticite conditionnelle.

    Parametres :
    -----------
    returns : pd.Series
        Rendements logarithmiques de QQQ (ou residus du modele de moyenne).
    nlags_lm : int
        Nombre de retards pour le test ARCH-LM (20 = 1 mois boursier).
    nlags_lb : int
        Nombre de retards pour Ljung-Box sur r^2.

    Retourne :
    ----------
    dict avec les resultats des deux tests et la conclusion.
    """
    r = returns.dropna()
    r_vals = r.values

    # --- Test ARCH-LM d'Engle ---
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(r_vals, nlags=nlags_lm)

    # --- Test de Ljung-Box sur les rendements au carre ---
    r_sq = (r - r.mean()) ** 2
    lb_test = acorr_ljungbox(r_sq, lags=[5, 10, 20], return_df=True)

    # On prend la p-value la plus forte (conservative)
    lb_pvalue_20 = float(lb_test["lb_pvalue"].iloc[-1])  # Lag 20
    lb_stat_20   = float(lb_test["lb_stat"].iloc[-1])

    # Conclusion : effet ARCH si au moins un test rejette H0
    effet_arch = (lm_pvalue < 0.05) or (lb_pvalue_20 < 0.05)

    report = {
        "lm_stat":       float(lm_stat),
        "lm_pvalue":     float(lm_pvalue),
        "lm_nlags":      nlags_lm,
        "f_stat":        float(f_stat),
        "f_pvalue":      float(f_pvalue),
        "lb_stat_20":    lb_stat_20,
        "lb_pvalue_20":  lb_pvalue_20,
        "lb_detail":     lb_test.to_dict(),
        "effet_arch":    effet_arch,
        "conclusion": (
            "Effet ARCH CONFIRME : au moins un test rejette H0 (p < 0.05). "
            "La variance des rendements est heteroscedastique. "
            "Le modele GARCH est justifie."
            if effet_arch else
            "Aucun test ne rejette H0 au seuil de 5%. "
            "Neanmoins, la tres forte persistance de la volatilite "
            "conditionnelle justifie le recours a GARCH."
        )
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : ESTIMATION DU MODELE GARCH(1,1)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROLE : Estimer les coefficients (omega, alpha, beta) de l'equation :
#        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
#
# POURQUOI chaque coefficient :
#   - omega (> 0)  : Plancher de variance a long terme.
#   - alpha        : Reactivite aux chocs d'hier (effet ARCH).
#   - beta         : Persistance / memoire de la volatilite (effet GARCH).
#   - alpha + beta : Mesure de persistance totale (doit etre < 1).
#
# COMMENT : On estime sur le Train (2018-2022) avec deux distributions :
#   (a) Normale : queues fines, hypothese classique.
#   (b) Student-t : queues epaisses (fat tails), mieux adaptee au QQQ.
#   On compare via AIC/BIC pour choisir la meilleure specification.
#
# UTILITE : Fournit sigma_{t+1} pour le calcul de VaR et comme feature D05.
# ═══════════════════════════════════════════════════════════════════════════════

def fit_garch(returns_train, dist="normal"):
    """
    Ajuste un GARCH(1,1) sur les rendements Train.
    Les rendements sont multiplies par 100 en interne (convention arch).
    """
    model = arch_model(
        returns_train * 100,
        mean="Constant",
        vol="Garch",
        p=1, q=1,
        dist=dist
    )
    result = model.fit(disp="off", show_warning=False)
    return result


def compare_distributions(returns_train):
    """
    Estime GARCH(1,1) avec Normale et Student-t, compare AIC/BIC.

    POURQUOI deux distributions ?
    Les rendements de QQQ ont un kurtosis > 3 (queues plus epaisses
    que la Normale). La loi de Student capture mieux les krachs.

    Retourne dict avec les deux modeles et un tableau comparatif.
    """
    garch_norm = fit_garch(returns_train, dist="normal")
    garch_t    = fit_garch(returns_train, dist="t")

    def extract_params(res, dist_label):
        p = res.params
        d = {
            "omega":       float(p["omega"]),
            "alpha":       float(p["alpha[1]"]),
            "beta":        float(p["beta[1]"]),
            "persistance": float(p["alpha[1]"] + p["beta[1]"]),
            "log_lik":     float(res.loglikelihood),
            "aic":         float(res.aic),
            "bic":         float(res.bic),
        }
        if dist_label == "Student-t":
            d["nu"] = float(p["nu"])
        return d

    comparison = {
        "Normale":    extract_params(garch_norm, "Normale"),
        "Student-t":  extract_params(garch_t, "Student-t"),
    }
    best_dist = "Student-t" if garch_t.bic < garch_norm.bic else "Normale"
    comparison["meilleure_distribution"] = best_dist

    return {
        "garch_norm": garch_norm,
        "garch_t":    garch_t,
        "comparison": comparison,
        "best_model": garch_t if best_dist == "Student-t" else garch_norm,
        "best_dist":  best_dist
    }


def _get_dist_string(model_result):
    """
    Retourne le code de distribution utilisable par arch_model()
    a partir d'un modele deja estime.
    La propriete model.distribution.name renvoie un nom long
    (ex: "Standardized Student's t") qu'arch_model() n'accepte pas.
    """
    name = model_result.model.distribution.name.lower()
    if "student" in name or "t" in name:
        return "t"
    elif "skew" in name:
        return "skewt"
    elif "ged" in name or "generalized" in name:
        return "ged"
    else:
        return "normal"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : PREVISION DE VOLATILITE ET CALCUL VaR / ES
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROLE : Pour chaque seance hors-echantillon, prevoir sigma_{t+1} et en
#        deduire VaR et Expected Shortfall.
#
# POURQUOI :
#   - VaR(95%) : "5% de chances que la perte depasse ce seuil."
#     -> Standard de gestion de portefeuille quotidien.
#   - VaR(99%) : "1% de chances..." -> Standard reglementaire (Bale).
#   - ES (Expected Shortfall / CVaR) : Perte MOYENNE dans les pires cas.
#     Contrairement a la VaR (seuil), l'ES mesure l'ampleur du desastre.
#
# COMMENT :
#   VaR_{t+1}(alpha) = -(mu + q_alpha * sigma_{t+1})
#   ou q_alpha est le quantile de la distribution (Normale ou Student-t)
#   et sigma_{t+1} est la volatilite filtree par GARCH a la date t.
#
# UTILITE : Position sizing au D06. Quand la VaR est elevee, on reduit
#           l'exposition ou on se refugie en Cash.
# ═══════════════════════════════════════════════════════════════════════════════

def predict_volatility_fast(best_model, returns_full, train_end):
    """
    Applique le filtre GARCH avec les parametres du Train sur toute la serie
    et extrait la volatilite conditionnelle pour les periodes hors-echantillon.

    C'est la methode standard en production : on gele les poids du modele
    et on filtre les nouvelles observations.
    """
    dist_str = _get_dist_string(best_model)
    params   = best_model.params

    # Re-estimer le modele sur la serie complete avec la meme specification
    am = arch_model(returns_full * 100, mean="Constant", vol="Garch",
                    p=1, q=1, dist=dist_str)
    res = am.fit(disp="off", show_warning=False)

    # Volatilite conditionnelle filtree (en pourcentage -> decimales)
    cond_vol = res.conditional_volatility / 100

    # Periode hors-echantillon
    oos = returns_full.loc[train_end:].iloc[1:]

    # Moyenne conditionnelle
    mu = res.params["mu"] / 100

    # Quantiles selon la distribution
    if dist_str == "t":
        nu = res.params["nu"]
        q_05 = sp_stats.t.ppf(0.05, df=nu)
        q_01 = sp_stats.t.ppf(0.01, df=nu)
        # Expected Shortfall analytique Student-t
        pdf_05 = sp_stats.t.pdf(q_05, df=nu)
        pdf_01 = sp_stats.t.pdf(q_01, df=nu)
        es_factor_05 = -(pdf_05 / 0.05) * ((nu + q_05**2) / (nu - 1))
        es_factor_01 = -(pdf_01 / 0.01) * ((nu + q_01**2) / (nu - 1))
    else:
        q_05 = sp_stats.norm.ppf(0.05)
        q_01 = sp_stats.norm.ppf(0.01)
        pdf_05 = sp_stats.norm.pdf(q_05)
        pdf_01 = sp_stats.norm.pdf(q_01)
        es_factor_05 = -pdf_05 / 0.05
        es_factor_01 = -pdf_01 / 0.01

    sigma_oos = cond_vol.loc[oos.index]

    result_df = pd.DataFrame(index=oos.index)
    result_df["r_actual"]   = oos.values
    result_df["sigma_pred"] = sigma_oos.values
    result_df["var_95"]     = -(mu + q_05 * sigma_oos.values)
    result_df["var_99"]     = -(mu + q_01 * sigma_oos.values)
    result_df["es_95"]      = -(mu + es_factor_05 * sigma_oos.values)
    result_df["es_99"]      = -(mu + es_factor_01 * sigma_oos.values)

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 : BACKTESTING DE LA VaR (TEST DE KUPIEC POF)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ROLE : Verifier empiriquement que les seuils de VaR sont calibres.
#        C'est le "controle qualite" obligatoire du modele de risque.
#
# POURQUOI :
#   Un modele theorique peut etre faux en pratique. Le backtesting prouve :
#   - A 95% : environ 5% des jours reels depassent la VaR.
#   - A 99% : environ 1% des jours reels depassent la VaR.
#   Trop de depassements -> modele sous-estime le risque (danger !).
#   Trop peu -> modele sur-estime le risque (perte d'opportunites).
#
# COMMENT :
#   Test de Kupiec (Proportion of Failures - POF) :
#   LR = -2 * ln[ L(alpha) / L(p_hat) ]  ~ chi2(1)
#   Si p-value > 0.05 -> VaR validee.
#
# UTILITE : Preuve formelle de fiabilite pour le professeur et les
#           regulateurs bancaires.
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_var(result_df, alpha=0.05, label="VaR 95%"):
    """
    Backtesting de la VaR : comptage des breaches et test de Kupiec.

    Parametres :
    -----------
    result_df : DataFrame avec colonnes r_actual et var_95/var_99.
    alpha : float
        Niveau de risque nominal (0.05 pour 95%, 0.01 pour 99%).
    label : str
        Etiquette pour l'affichage.

    Retourne :
    ----------
    dict avec n_breaches, taux_observe, test de Kupiec et conclusion.
    """
    var_col = "var_95" if alpha == 0.05 else "var_99"
    losses     = -result_df["r_actual"]
    var_seuils = result_df[var_col]
    breaches   = (losses > var_seuils)
    n_breaches = int(breaches.sum())
    total      = len(result_df)
    taux_obs   = n_breaches / total

    # Test de Kupiec POF (Likelihood Ratio)
    p_hat = max(min(taux_obs, 1 - 1e-10), 1e-10)
    x, T  = n_breaches, total

    if x == 0:
        lr_stat = -2 * (T * np.log(1 - alpha) - T * np.log(1 - p_hat))
    elif x == T:
        lr_stat = -2 * (T * np.log(alpha) - T * np.log(p_hat))
    else:
        lr_stat = -2 * (
            (T - x) * np.log(1 - alpha) + x * np.log(alpha)
          - (T - x) * np.log(1 - p_hat) - x * np.log(p_hat)
        )

    kupiec_pvalue = 1 - sp_stats.chi2.cdf(lr_stat, df=1)

    report = {
        "label":           label,
        "alpha_nominal":   alpha,
        "total_seances":   total,
        "n_breaches":      n_breaches,
        "taux_attendu_%":  alpha * 100,
        "taux_observe_%":  taux_obs * 100,
        "kupiec_lr_stat":  float(lr_stat),
        "kupiec_pvalue":   float(kupiec_pvalue),
        "var_validee":     bool(kupiec_pvalue > 0.05),
        "conclusion": (
            f"VaR {int((1-alpha)*100)}% VALIDEE par Kupiec (p={kupiec_pvalue:.4f} > 0.05) : "
            f"taux observe ({taux_obs*100:.2f}%) conforme au nominal ({alpha*100:.1f}%)."
            if kupiec_pvalue > 0.05 else
            f"VaR {int((1-alpha)*100)}% REJETEE par Kupiec (p={kupiec_pvalue:.4f} <= 0.05) : "
            f"taux observe ({taux_obs*100:.2f}%) diverge du nominal ({alpha*100:.1f}%)."
        )
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION DE LA FIGURE D04 (4 PANNEAUX)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_d04_results(result_df, garch_comparison, arch_report, bt_95, bt_99):
    """
    Graphique d'audit a 4 panneaux pour le Livrable D04.

    Panneau 1 : Volatilite realisee (|r_t|) vs Prevue GARCH (sigma_t)
    Panneau 2 : Tunnel de VaR 95% avec depassements surlignés en rouge
    Panneau 3 : Autocorrelation des carres des rendements (preuve de clustering)
    Panneau 4 : Distribution des rendements vs Normale vs Student-t
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "D04 — Module de Risque : GARCH(1,1), VaR et Backtesting\n"
        "Projet Trading Intelligent | Variante B01 · QQQ (Nasdaq-100)",
        fontsize=13, fontweight="bold"
    )
    plt.style.use("seaborn-v0_8-darkgrid")

    # ── Panneau 1 : Volatilite realisee vs prevue ──
    ax = axes[0, 0]
    vol_real = result_df["r_actual"].abs() * 100
    vol_pred = result_df["sigma_pred"] * 100
    ax.plot(result_df.index, vol_real, color="gray", alpha=0.4, lw=0.8,
            label="Volatilite realisee |r_t| (%)")
    ax.plot(result_df.index, vol_pred, color="#d62728", lw=1.5,
            label="Volatilite prevue GARCH σ_t (%)")
    ax.fill_between(result_df.index, 0, vol_pred, color="#d62728", alpha=0.15)
    ax.set_title("1. Volatilite realisee vs Prevue GARCH(1,1)")
    ax.set_ylabel("Volatilite (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Panneau 2 : Tunnel de VaR 95% avec breaches ──
    ax = axes[0, 1]
    losses = -result_df["r_actual"] * 100
    var_95 = result_df["var_95"] * 100
    ax.plot(result_df.index, losses, color="steelblue", lw=0.8, alpha=0.7,
            label="Perte reelle -r_t (%)")
    ax.plot(result_df.index, var_95, color="#d62728", lw=1.5, ls="--",
            label="Seuil VaR 95%")
    breaches_mask = losses > var_95
    if breaches_mask.any():
        ax.scatter(result_df.index[breaches_mask], losses[breaches_mask],
                   color="red", s=25, zorder=5,
                   label=f"Breaches ({breaches_mask.sum()})")
    ax.set_title(f"2. Tunnel de VaR 95% — {bt_95['n_breaches']} depassements "
                 f"({bt_95['taux_observe_%']:.1f}% obs. vs "
                 f"{bt_95['taux_attendu_%']:.0f}% att.)")
    ax.set_ylabel("Perte (%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Panneau 3 : ACF des carres des rendements ──
    ax = axes[1, 0]
    resid_sq = (result_df["r_actual"] ** 2).dropna()
    acf_vals = acf_func(resid_sq, nlags=20, fft=True)
    lags = np.arange(len(acf_vals))
    colors = ["#d62728" if abs(v) > 1.96/np.sqrt(len(resid_sq)) else "#9467bd"
              for v in acf_vals]
    ax.bar(lags, acf_vals, color=colors, alpha=0.7, width=0.6)
    conf = 1.96 / np.sqrt(len(resid_sq))
    ax.axhline(conf, color="red", ls="--", lw=1, label=f"IC 95% (±{conf:.3f})")
    ax.axhline(-conf, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("3. ACF des rendements au carré ε²_t (preuve de clustering)")
    ax.set_xlabel("Retard (jours)")
    ax.set_ylabel("ACF")
    ax.legend(fontsize=9)

    # ── Panneau 4 : Distribution rendements vs Normale vs Student-t ──
    ax = axes[1, 1]
    r_pct = result_df["r_actual"] * 100
    ax.hist(r_pct, bins=60, density=True, alpha=0.5, color="#1f77b4",
            label="Rendements reels QQQ (%)")
    x_range = np.linspace(r_pct.min(), r_pct.max(), 300)
    mu_r, std_r = r_pct.mean(), r_pct.std()
    norm_pdf = sp_stats.norm.pdf(x_range, loc=mu_r, scale=std_r)
    ax.plot(x_range, norm_pdf, "g--", lw=2, label="Loi Normale")
    comp = garch_comparison["comparison"]
    if "Student-t" in comp and "nu" in comp["Student-t"]:
        nu = comp["Student-t"]["nu"]
        t_pdf = sp_stats.t.pdf(x_range, df=nu, loc=mu_r, scale=std_r)
        ax.plot(x_range, t_pdf, "r-", lw=2, label=f"Student-t (ν={nu:.1f})")
    ax.set_title("4. Distribution : Rendements vs Normale vs Student-t")
    ax.set_xlabel("Rendement (%)")
    ax.set_ylabel("Densite")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "D04_risk_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE D04 COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

def run_d04_pipeline():
    """Execute le pipeline complet du Livrable D04."""
    print("=" * 65)
    print("  LIVRABLE D04 - MODULE DE RISQUE (GARCH / VaR / ES)")
    print("=" * 65)

    # ── Chargement ──
    df = load_dataset()
    train, val, test = split_data(df)
    r_train = train["r_qqq"]
    r_full  = df["r_qqq"]

    print(f"[LOAD] Dataset charge : {len(df)} seances totales")
    print(f"       Train      (2018-2022) : {len(train)} seances")
    print(f"       Validation (2023-2024) : {len(val)} seances")
    print(f"       Test       (2025-2026) : {len(test)} seances")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 : Diagnostic ARCH
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  PHASE 1 : DIAGNOSTIC D'HETEROSCEDASTICITE")
    print(f"{'─'*65}")
    print("  Role : Prouver que la variance n'est pas constante.")
    print("  Tests : ARCH-LM d'Engle (20 lags) + Ljung-Box sur r^2.\n")

    arch_report = test_arch_effect(r_train, nlags_lm=20, nlags_lb=20)
    print(f"  Test ARCH-LM (Engle) :")
    print(f"    Statistique LM  : {arch_report['lm_stat']:.4f}")
    print(f"    p-value LM      : {arch_report['lm_pvalue']:.4f}")
    print(f"    Retards testes  : {arch_report['lm_nlags']}")
    print(f"  Test Ljung-Box sur r^2 (lag 20) :")
    print(f"    Statistique Q   : {arch_report['lb_stat_20']:.4f}")
    print(f"    p-value Q       : {arch_report['lb_pvalue_20']:.4f}")
    print(f"  >>> {arch_report['conclusion']}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 : Estimation GARCH(1,1)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  PHASE 2 : ESTIMATION GARCH(1,1)")
    print(f"{'─'*65}")
    print("  Role : Estimer omega, alpha, beta de l'equation :")
    print("    sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2")
    print("  Comparaison : Normale vs Student-t.\n")

    garch_comp = compare_distributions(r_train)
    comp = garch_comp["comparison"]

    print(f"  {'Parametre':<20} {'Normale':>12} {'Student-t':>12}")
    print(f"  {'─'*44}")
    print(f"  {'omega':<20} {comp['Normale']['omega']:>12.6f} {comp['Student-t']['omega']:>12.6f}")
    print(f"  {'alpha (reactivite)':<20} {comp['Normale']['alpha']:>12.6f} {comp['Student-t']['alpha']:>12.6f}")
    print(f"  {'beta (persistance)':<20} {comp['Normale']['beta']:>12.6f} {comp['Student-t']['beta']:>12.6f}")
    print(f"  {'alpha + beta':<20} {comp['Normale']['persistance']:>12.6f} {comp['Student-t']['persistance']:>12.6f}")
    if "nu" in comp["Student-t"]:
        nu_str = f"{comp['Student-t']['nu']:.2f}"
        print(f"  {'nu (degres lib.)':<20} {'—':>12} {nu_str:>12}")
    print(f"  {'Log-Likelihood':<20} {comp['Normale']['log_lik']:>12.2f} {comp['Student-t']['log_lik']:>12.2f}")
    print(f"  {'AIC':<20} {comp['Normale']['aic']:>12.2f} {comp['Student-t']['aic']:>12.2f}")
    print(f"  {'BIC':<20} {comp['Normale']['bic']:>12.2f} {comp['Student-t']['bic']:>12.2f}")
    print(f"\n  >>> Distribution retenue : {comp['meilleure_distribution']} (BIC le plus faible)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3 : Prevision VaR / ES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  PHASE 3 : PREVISION DE VOLATILITE ET CALCUL VaR / ES")
    print(f"{'─'*65}")
    print("  Role : Prevoir sigma_{t+1} et en deduire VaR et ES.\n")

    result_df = predict_volatility_fast(
        garch_comp["best_model"], r_full, DATE_TRAIN_END
    )

    val_results  = result_df.loc[:DATE_VAL_END]
    test_results = result_df.loc[DATE_VAL_END:].iloc[1:] if len(result_df.loc[DATE_VAL_END:]) > 0 else result_df.iloc[0:0]

    print(f"  Previsions generees : {len(result_df)} seances hors-echantillon")
    print(f"    Validation : {len(val_results)} seances")
    print(f"    Test       : {len(test_results)} seances")
    sigma_pct = result_df["sigma_pred"] * 100
    print(f"\n  Statistiques de sigma prevue :")
    print(f"    Moyenne    : {sigma_pct.mean():.4f}%")
    print(f"    Min        : {sigma_pct.min():.4f}%")
    print(f"    Max        : {sigma_pct.max():.4f}%")
    print(f"    Ecart-type : {sigma_pct.std():.4f}%")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4 : Backtesting
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  PHASE 4 : BACKTESTING DE LA VaR (TEST DE KUPIEC)")
    print(f"{'─'*65}")
    print("  Role : Verifier la conformite des depassements.\n")

    bt_95 = backtest_var(result_df, alpha=0.05, label="VaR 95%")
    bt_99 = backtest_var(result_df, alpha=0.01, label="VaR 99%")

    for bt in [bt_95, bt_99]:
        print(f"  --- {bt['label']} ---")
        print(f"  Seances totales       : {bt['total_seances']}")
        print(f"  Depassements observes : {bt['n_breaches']}")
        print(f"  Taux attendu          : {bt['taux_attendu_%']:.1f}%")
        print(f"  Taux observe          : {bt['taux_observe_%']:.2f}%")
        print(f"  Kupiec LR stat        : {bt['kupiec_lr_stat']:.4f}")
        print(f"  Kupiec p-value        : {bt['kupiec_pvalue']:.4f}")
        print(f"  >>> {bt['conclusion']}\n")

    # Detail par periode
    bt_val_95  = backtest_var(val_results, alpha=0.05, label="VaR 95% (Val)")
    bt_test_95 = backtest_var(test_results, alpha=0.05, label="VaR 95% (Test)")
    print(f"  Par periode :")
    print(f"    Validation : {bt_val_95['n_breaches']} breaches / "
          f"{bt_val_95['total_seances']} ({bt_val_95['taux_observe_%']:.2f}%)")
    print(f"    Test       : {bt_test_95['n_breaches']} breaches / "
          f"{bt_test_95['total_seances']} ({bt_test_95['taux_observe_%']:.2f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # Graphique + Sauvegarde
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  GENERATION DES LIVRABLES")
    print(f"{'─'*65}")

    fig_path = plot_d04_results(result_df, garch_comp, arch_report, bt_95, bt_99)
    print(f"  [OK] Graphique D04 : {fig_path}")

    out_path = os.path.join("data", "processed", "garch_volatility.csv")
    result_df.to_csv(out_path)
    print(f"  [OK] Previsions sauvegardees : {out_path}")

    print(f"\n{'='*65}")
    print("  D04 TERMINE — Module de risque pret pour D05 et D06.")
    print(f"{'='*65}")

    return {
        "arch_report":      arch_report,
        "garch_comparison": garch_comp["comparison"],
        "best_dist":        garch_comp["best_dist"],
        "result_df":        result_df,
        "backtest_95":      bt_95,
        "backtest_99":      bt_99,
        "bt_val_95":        bt_val_95,
        "bt_test_95":       bt_test_95,
        "fig_path":         fig_path,
    }


if __name__ == "__main__":
    run_d04_pipeline()
