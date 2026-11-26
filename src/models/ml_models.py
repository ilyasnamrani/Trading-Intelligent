"""
D05 - Module de Machine Learning et Deep Learning
Projet : Trading Intelligent - Variante B01 (QQQ vs SPY)

Objectif :
----------
Construire, optimiser et comparer des modeles d'apprentissage supervise
(ML et Deep Learning) pour predire la direction du rendement de QQQ a t+1 :
  1. Feature Engineering avance (Technique + Exogene SPY + Volatilite GARCH)
  2. Modeles etudies :
     - Baseline Naive (Majoritaire / Random)
     - Modele Lineaire Regularise : Logistic Regression L2 (Ridge)
     - Modeles Arborescents : Random Forest & HistGradientBoosting
     - Modele Deep Learning : Multi-Layer Perceptron (MLP) compact
  3. Protocole temporel strict sans fuite d'information (no lookahead bias) :
     - Train : 2018-2022 (calibration des poids et du scaler)
     - Validation : 2023-2024 (tuning des hyperparametres)
     - Test Gelé : 2025-2026 (evaluation aveugle finale)
  4. Comparaison directe avec le benchmark SARIMAX(1,0,0) du D03
  5. Sauvegarde des signaux predictifs pour la strategie du D06
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─── CONFIGURATION & CHEMINS ────────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
GARCH_DATA_PATH     = os.path.join("data", "processed", "garch_volatility.csv")
FIGURES_DIR         = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATE_TRAIN_END = "2022-12-31"
DATE_VAL_END   = "2024-12-31"
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET INGENIERIE DES FEATURES D05
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rsi(prices, window=14):
    """Calcule le Relative Strength Index (RSI) sur 14 seances."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD (Moving Average Convergence Divergence) et sa ligne de signal."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def build_ml_dataset():
    """
    Construit la matrice de features complete et la variable cible binaire.
    Fusionne les donnees de marche (D02) et la volatilite conditionnelle GARCH (D04).
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Fichier {PROCESSED_DATA_PATH} introuvable. Executez src/data.py.")
    
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Features de rendement retardes QQQ
    df["r_qqq_lag2"] = df["r_qqq"].shift(2)
    df["r_qqq_lag3"] = df["r_qqq"].shift(3)
    
    # 2. Features de benchmark retardes SPY
    df["r_spy_lag2"] = df["r_spy"].shift(2)
    
    # 3. Indicateurs techniques avances (connus a la cloture t)
    df["rsi_14"] = compute_rsi(df["close_qqq"], window=14)
    df["sma_20"] = df["close_qqq"].rolling(20).mean()
    df["sma_ratio"] = df["close_qqq"] / df["sma_20"] - 1.0  # % d'ecart a la SMA20
    
    macd_l, macd_s, macd_h = compute_macd(df["close_qqq"])
    df["macd_line"] = macd_l
    df["macd_diff"] = macd_h
    
    # 4. Volatilite conditionnelle GARCH (Livrable D04)
    # Si disponible, on injecte sigma_pred, sinon on utilise la volatilite realisee 20j
    if os.path.exists(GARCH_DATA_PATH):
        garch_df = pd.read_csv(GARCH_DATA_PATH, index_col=0, parse_dates=True)
        # Aligner sigma_pred
        df["garch_vol"] = garch_df["sigma_pred"].reindex(df.index)
        # Remplir le Train (ou sigma_pred n'a pas ete exporte separement) par vol_20
        df["garch_vol"] = df["garch_vol"].fillna(df["vol_20"])
    else:
        df["garch_vol"] = df["vol_20"]

    # 5. Cible binaire : Direction de demain (1 si r_qqq[t+1] > 0, sinon 0)
    # target = r_qqq[t+1] est deja calcule au D02
    df["target_direction"] = (df["target"] > 0).astype(int)
    
    # Selection stricte des features d'entree (connues a t, pas de fuite)
    feature_cols = [
        "r_qqq",          # Rendement du jour
        "r_qqq_lag1",     # Rendement t-1
        "r_qqq_lag2",     # Rendement t-2
        "r_qqq_lag5",     # Rendement t-5 (effet semaine)
        "r_spy",          # Rendement SPY du jour
        "r_spy_lag1",     # Rendement SPY t-1
        "vol_20",         # Volatilite realisee 20j
        "garch_vol",      # Volatilite conditionnelle GARCH
        "rsi_14",         # RSI 14 jours
        "sma_ratio",      # Ratio Prix / SMA20
        "macd_diff",      # Histogramme MACD
    ]
    
    clean_df = df.dropna(subset=feature_cols + ["target_direction", "target"]).copy()
    return clean_df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DECOUPAGE TEMPOREL ET STANDARDISATION ETANCHE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_splits(df, feature_cols):
    """
    Decoupe selon les 3 fenetres temporelles etanches et applique
    un StandardScaler ajuste UNIQUEMENT sur le Train (zero fuite).
    """
    train_mask = df.index <= DATE_TRAIN_END
    val_mask   = (df.index > DATE_TRAIN_END) & (df.index <= DATE_VAL_END)
    test_mask  = df.index > DATE_VAL_END
    
    X = df[feature_cols]
    y = df["target_direction"]
    r_target = df["target"]  # Rendement continu pour calcul financier
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    
    # Standardisation : fit sur Train UNIQUEMENT
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    splits = {
        "X_train": X_train_scaled, "y_train": y_train.values, "r_train": r_target[train_mask].values, "idx_train": X_train.index,
        "X_val":   X_val_scaled,   "y_val":   y_val.values,   "r_val":   r_target[val_mask].values,   "idx_val":   X_val.index,
        "X_test":  X_test_scaled,  "y_test":  y_test.values,  "r_test":  r_target[test_mask].values,  "idx_test":  X_test.index,
        "scaler": scaler,
        "raw_X": X
    }
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DEFINITION ET ENTRAINEMENT DES MODELES
# ═══════════════════════════════════════════════════════════════════════════════

def get_models():
    """
    Retourne le dictionnaire des modeles a calibrer, couvrant
    les 3 familles du cahier des charges :
      1. Lineaire regularise (Logistic Regression L2 / Ridge)
      2. Modeles Arborescents (Random Forest, HistGradientBoosting)
      3. Deep Learning (MLP compact)
    """
    models = {
        # 1. Famille Lineaire Regularisee (Ridge / Logistic L2)
        "Logistic_L2_Ridge": LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", random_state=SEED, max_iter=500
        ),
        
        # 2. Famille Arborescente : Random Forest
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_split=10,
            max_features="sqrt", random_state=SEED, n_jobs=-1
        ),
        
        # 2. Famille Arborescente : Gradient Boosting
        "Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, learning_rate=0.03,
            l2_regularization=1.0, random_state=SEED
        ),
        
        # 3. Famille Deep Learning : MLP Compact
        "MLP_Neural_Net": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,  # Regularisation L2 (Weight Decay)
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED
        )
    }
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION ET BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_single_model(model, X_train, y_train, X_eval, y_eval, model_name="Model"):
    """Entraine un modele et calcule ses metriques completes."""
    # Temps d'entrainement
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # Temps d'inference et predictions
    t1 = time.time()
    y_pred = model.predict(X_eval)
    pred_time = (time.time() - t1) / len(X_eval) * 1000  # ms par observation
    
    # Probabilites (pour ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = y_pred
    
    acc  = accuracy_score(y_eval, y_pred) * 100.0
    prec = precision_score(y_eval, y_pred, zero_division=0) * 100.0
    rec  = recall_score(y_eval, y_pred, zero_division=0) * 100.0
    f1   = f1_score(y_eval, y_pred, average="macro") * 100.0
    
    try:
        auc = roc_auc_score(y_eval, y_prob) * 100.0
    except Exception:
        auc = 50.0
        
    return {
        "Modele": model_name,
        "Accuracy_%": acc,
        "Hit_Ratio_%": acc,  # Identique a l'Accuracy sur la classification de direction
        "Precision_%": prec,
        "Recall_%": rec,
        "Macro_F1_%": f1,
        "ROC_AUC_%": auc,
        "Fit_Time_s": fit_time,
        "Pred_Time_ms": pred_time,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def run_benchmark(splits, models):
    """Evalue l'ensemble des modeles sur Validation et Test."""
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_v,  y_v  = splits["X_val"],   splits["y_val"]
    X_te, y_te = splits["X_test"],  splits["y_test"]
    
    # Baseline Naive (Majoritaire sur Train)
    majority_class = int(y_tr.mean() >= 0.5)
    y_pred_base_v  = np.full_like(y_v, majority_class)
    y_pred_base_te = np.full_like(y_te, majority_class)
    
    base_val = {
        "Modele": "Baseline_Majoritaire",
        "Accuracy_%": (y_v == y_pred_base_v).mean() * 100.0,
        "Hit_Ratio_%": (y_v == y_pred_base_v).mean() * 100.0,
        "Precision_%": precision_score(y_v, y_pred_base_v, zero_division=0) * 100.0,
        "Recall_%": recall_score(y_v, y_pred_base_v, zero_division=0) * 100.0,
        "Macro_F1_%": f1_score(y_v, y_pred_base_v, average="macro") * 100.0,
        "ROC_AUC_%": 50.0,
        "Fit_Time_s": 0.001,
        "Pred_Time_ms": 0.001,
        "y_pred": y_pred_base_v,
        "y_prob": np.full_like(y_v, 0.5, dtype=float)
    }
    
    base_test = {
        "Modele": "Baseline_Majoritaire",
        "Accuracy_%": (y_te == y_pred_base_te).mean() * 100.0,
        "Hit_Ratio_%": (y_te == y_pred_base_te).mean() * 100.0,
        "Precision_%": precision_score(y_te, y_pred_base_te, zero_division=0) * 100.0,
        "Recall_%": recall_score(y_te, y_pred_base_te, zero_division=0) * 100.0,
        "Macro_F1_%": f1_score(y_te, y_pred_base_te, average="macro") * 100.0,
        "ROC_AUC_%": 50.0,
        "Fit_Time_s": 0.001,
        "Pred_Time_ms": 0.001,
        "y_pred": y_pred_base_te,
        "y_prob": np.full_like(y_te, 0.5, dtype=float)
    }
    
    val_metrics  = [base_val]
    test_metrics = [base_test]
    eval_cache   = {}
    
    for name, model in models.items():
        # Fit une fois sur Train, evalue sur Val
        res_v = evaluate_single_model(model, X_tr, y_tr, X_v, y_v, model_name=name)
        val_metrics.append(res_v)
        
        # Evalue sur Test (sans refit pour tester la robustesse temporelle pure)
        t1 = time.time()
        y_pred_te = model.predict(X_te)
        pred_time_te = (time.time() - t1) / len(X_te) * 1000
        y_prob_te = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred_te
        
        acc_te  = accuracy_score(y_te, y_pred_te) * 100.0
        prec_te = precision_score(y_te, y_pred_te, zero_division=0) * 100.0
        rec_te  = recall_score(y_te, y_pred_te, zero_division=0) * 100.0
        f1_te   = f1_score(y_te, y_pred_te, average="macro") * 100.0
        auc_te  = roc_auc_score(y_te, y_prob_te) * 100.0 if hasattr(model, "predict_proba") else 50.0
        
        res_te = {
            "Modele": name,
            "Accuracy_%": acc_te,
            "Hit_Ratio_%": acc_te,
            "Precision_%": prec_te,
            "Recall_%": rec_te,
            "Macro_F1_%": f1_te,
            "ROC_AUC_%": auc_te,
            "Fit_Time_s": res_v["Fit_Time_s"],
            "Pred_Time_ms": pred_time_te,
            "y_pred": y_pred_te,
            "y_prob": y_prob_te
        }
        test_metrics.append(res_te)
        eval_cache[name] = {"val": res_v, "test": res_te, "model": model}
        
    df_val  = pd.DataFrame(val_metrics).drop(columns=["y_pred", "y_prob"])
    df_test = pd.DataFrame(test_metrics).drop(columns=["y_pred", "y_prob"])
    
    return df_val, df_test, eval_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION D'AUDIT D05 (4 PANNEAUX)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_d05_results(df_val, df_test, eval_cache, feature_names, splits):
    """
    Genere la figure consolidée a 4 panneaux pour le Livrable D05 :
      1. Comparatif Hit Ratio / Accuracy (Validation vs Test)
      2. Importance des features (Random Forest)
      3. Courbes ROC sur l'ensemble Test
      4. Matrice de Confusion du Modele Champion
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "D05 — Machine Learning et Deep Learning : Benchmark Predictif de Direction\n"
        "Projet Trading Intelligent | Variante B01 · QQQ (Nasdaq-100)",
        fontsize=13, fontweight="bold"
    )
    plt.style.use("seaborn-v0_8-darkgrid")
    
    # ── Panneau 1 : Barplot Comparatif Accuracy / Hit Ratio ──
    ax = axes[0, 0]
    models_list = df_val["Modele"].tolist()
    x = np.arange(len(models_list))
    width = 0.35
    
    val_acc  = df_val["Accuracy_%"].values
    test_acc = df_test["Accuracy_%"].values
    
    rects1 = ax.bar(x - width/2, val_acc, width, label="Validation (2023–2024)", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x + width/2, test_acc, width, label="Test Gelé (2025–2026)", color="#2ca02c", alpha=0.85)
    
    # Ligne de reference 50%
    ax.axhline(50.0, color="red", linestyle="--", lw=1.2, label="Seuil Aléatoire (50%)")
    # Ligne SARIMAX D03 (Validation 52.78%, Test 50.31%)
    ax.axhline(52.78, color="#ff7f0e", linestyle=":", lw=1.5, label="Ref. SARIMAX Val (52.78%)")
    
    ax.set_title("1. Comparatif de Directional Accuracy / Hit Ratio (%)")
    ax.set_ylabel("Hit Ratio (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models_list], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(45, 60)
    ax.legend(fontsize=8, loc="upper right")
    
    # ── Panneau 2 : Importance des Features (Random Forest) ──
    ax = axes[0, 1]
    rf_model = eval_cache["Random_Forest"]["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    ax.barh(range(len(sorted_features)), sorted_importances, color="#9467bd", alpha=0.85)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_title("2. Importance des Variables (Random Forest Gini MDI)")
    ax.set_xlabel("Score d'Importance Relative")
    
    # ── Panneau 3 : Courbes ROC sur Test Gelé ──
    ax = axes[1, 0]
    y_test = splits["y_test"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    
    for (name, color) in zip(["Logistic_L2_Ridge", "Random_Forest", "Gradient_Boosting", "MLP_Neural_Net"], colors):
        y_prob = eval_cache[name]["test"]["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob) * 100.0
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name.replace('_', ' ')} (AUC = {auc:.1f}%)")
        
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Aléatoire (50%)")
    ax.set_title("3. Courbes ROC sur l'Échantillon Test Gelé (2025–2026)")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.legend(fontsize=8, loc="lower right")
    
    # ── Panneau 4 : Matrice de Confusion du Champion ──
    ax = axes[1, 1]
    # Selectionner le champion selon l'Accuracy sur Validation
    val_models_only = df_val[df_val["Modele"] != "Baseline_Majoritaire"]
    champion_name = val_models_only.sort_values("Accuracy_%", ascending=False).iloc[0]["Modele"]
    champion_pred = eval_cache[champion_name]["test"]["y_pred"]
    
    cm = confusion_matrix(y_test, champion_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baisse (0)", "Hausse (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"4. Matrice de Confusion Test : {champion_name.replace('_', ' ')}")
    ax.grid(False)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "D05_ml_dl_comparatif.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path, champion_name


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE D05 COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

def run_d05_pipeline():
    """Execute le pipeline complet du Livrable D05."""
    print("=" * 65)
    print("  LIVRABLE D05 - MACHINE LEARNING ET DEEP LEARNING")
    print("=" * 65)
    
    # 1. Dataset & Features
    df, feature_cols = build_ml_dataset()
    splits = prepare_splits(df, feature_cols)
    
    print(f"[1/4] Dataset ML construit : {len(df)} seances x {len(feature_cols)} features.")
    print(f"      Train      (2018-2022) : {len(splits['X_train'])} seances")
    print(f"      Validation (2023-2024) : {len(splits['X_val'])} seances")
    print(f"      Test Gelé  (2025-2026) : {len(splits['X_test'])} seances")
    print(f"      Features incluses : {', '.join(feature_cols)}")
    
    # 2. Modeles et Entrainement
    models = get_models()
    print(f"\n[2/4] Entrainement des 4 modeles supervises (Seed={SEED})...")
    df_val, df_test, eval_cache = run_benchmark(splits, models)
    
    # 3. Affichage des resultats
    print("\n" + "─" * 65)
    print("  TABLEAU COMPARATIF - VALIDATION (2023–2024)")
    print("─" * 65)
    print(df_val.to_string(index=False))
    
    print("\n" + "─" * 65)
    print("  TABLEAU COMPARATIF - TEST GELÉ (2025–2026)")
    print("─" * 65)
    print(df_test.to_string(index=False))
    
    # 4. Graphique consolidé et Modele Champion
    print(f"\n[3/4] Generation du graphique d'audit D05...")
    fig_path, champion_name = plot_d05_results(
        df_val, df_test, eval_cache, feature_cols, splits
    )
    print(f"      Graphique genere : {fig_path}")
    print(f"      >>> Modèle CHAMPION retenu : {champion_name}")
    
    # 5. Sauvegarde des signaux pour le backtesting D06
    print(f"\n[4/4] Sauvegarde des predictions pour le Livrable D06...")
    pred_export = pd.DataFrame(index=splits["idx_test"])
    pred_export["r_actual"] = splits["r_test"]
    pred_export["target_direction"] = splits["y_test"]
    
    for name in models.keys():
        pred_export[f"pred_{name}"] = eval_cache[name]["test"]["y_pred"]
        pred_export[f"prob_{name}"] = eval_cache[name]["test"]["y_prob"]
        
    out_preds = os.path.join("data", "processed", "ml_predictions.csv")
    pred_export.to_csv(out_preds)
    print(f"      Fichier de signaux sauvegarde : {out_preds}")
    
    print(f"\n{'='*65}")
    print("  D05 TERMINE — Modeles ML/DL prets pour le Backtest (D06).")
    print(f"{'='*65}")
    
    return {
        "df_val": df_val,
        "df_test": df_test,
        "champion_name": champion_name,
        "fig_path": fig_path,
        "eval_cache": eval_cache,
        "feature_cols": feature_cols
    }


if __name__ == "__main__":
    run_d05_pipeline()
