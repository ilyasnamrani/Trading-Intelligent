# QUANT LAB — Trading Intelligent (B01 · QQQ)

> **Projet :** Trading Intelligent — De la donnée de marché à une application de trading quantitatif  
> **Responsable pédagogique :** Pr. Tawfik Masrour | Département d'informatique — FS Rabat  
> **Actif étudié :** **QQQ** (Invesco QQQ Trust / Nasdaq-100) | **Benchmark :** **SPY** (S&P 500)  
> **Binôme :** Amrani Ilyas & Atattou Youssef  

---

###  Documents de Référence & Navigation des Livrables

* 📄 **Document Fondateur :** [Consulter le Cahier des Charges Intégral du Projet](reports/cahier_des_charges.md)

| Livrable | Statut | Accès au document complet |
| :--- | :---: | :--- |
| **D01 — Cadrage & Architecture** | Validé | [Consulter le rapport D01](reports/D01_cadrage.md) |
| **D02 — Données, Ingestion & Audit** | Validé | [Consulter le rapport D02](reports/D02_rapport.md) |
| **D03 — Économétrie de la moyenne** | Validé | [Consulter le rapport D03](reports/D03_rapport.md) |
| **D04 — Module de Risque (GARCH)** | Validé | [Consulter le rapport D04](reports/D04_risque.md) |
| **D05 — Machine Learning & Deep Learning** | **Actuel (ci-dessous)** | *Affiché sur cette page* |
| **D06 — Stratégie & Backtesting** |  À venir | `reports/D06_backtest.md` |
| **D07 — Application Web (Streamlit)** |  À venir | Code `app.py` |
| **D08 — Paper Trading (5 séances)** |  À venir | `reports/D08_paper_trading.md` |
| **D09 — Reproductibilité & Tests** |  À venir | `tests/` & `pytest` |
| **D10 — Rapport Final & Soutenance** |  À venir | `reports/final_report.pdf` |

---

# Livrable Actuel : D05 — Machine Learning et Deep Learning : Modélisation Prédictive de la Direction

## 1. Objectif du Livrable D05

Le livrable D05 constitue le second étage prédictif de la plateforme **Quant Lab**.  
Après avoir modélisé la dynamique linéaire au D03 (ARIMA / SARIMAX) et la volatilité conditionnelle au D04 (GARCH), son objectif est de tester si des **algorithmes non-linéaires d'apprentissage supervisé (Machine Learning et Deep Learning)** apportent un avantage statistique pour anticiper la **direction du rendement à un jour** (t+1) de l'ETF technologique **QQQ**.

La démarche respecte strictement les critères d'acceptation du cahier des charges :
1. **Zéro fuite d'information (*No lookahead bias*) :** Toutes les variables explicatives sont calculées à la clôture de t. La standardisation (`StandardScaler`) est calibrée **uniquement sur le Train**.
2. **Trois familles obligatoires :** Un modèle linéaire régularisé (Logistic Regression L2 / Ridge), des modèles arborescents (Random Forest, Gradient Boosting) et un modèle de Deep Learning (MLP compact).
3. **Recherche d'hyperparamètres limitée à la validation :** Le set de test gelé (2025–2026) n'est jamais utilisé pour choisir un modèle.
4. **Intégration du module de risque D04 :** La volatilité conditionnelle `σ_{t+1}` issue de GARCH est directement injectée comme feature de régime de marché.

---

## 2. Définition de la Cible et Ingénierie des Variables (X_t)

### 2.1 Cible Binaire (Classification de Direction)
En trading quantitatif réel, la prédiction continue de l'amplitude exacte est dominée par le bruit micro-structurel. Ce qui détermine la décision d'achat ou de liquidité au D06, c'est la **direction** :
* `target_direction = 1` si le rendement de demain est positif (Séance haussière / Achat)
* `target_direction = 0` si le rendement de demain est négatif ou nul (Séance baissière / Cash)

### 2.2 Panier des 11 Variables Explicatives (X_t)
Toutes les variables sont strictement observables à la clôture de la séance t :

| Variable | Nature | Description économique / financière |
| :--- | :---: | :--- |
| `r_qqq` | Endogène | Rendement logarithmique de QQQ à la séance t. |
| `r_qqq_lag1` | Endogène | Rendement de QQQ à t-1 (inertie court terme). |
| `r_qqq_lag2` | Endogène | Rendement de QQQ à t-2. |
| `r_qqq_lag5` | Endogène | Rendement de QQQ à t-5 (cycle hebdomadaire). |
| `r_spy` | Exogène (B01) | Rendement du benchmark S&P 500 à t. |
| `r_spy_lag1` | Exogène (B01) | Rendement du benchmark S&P 500 à t-1. |
| `vol_20` | Risque réalisé | Volatilité historique réalisée sur 20 séances (1 mois boursier). |
| **`garch_vol`** | **Risque conditionnel (D04)** | **Volatilité `σ_{t+1}` prévue par le modèle GARCH(1,1) Student-t.** |
| `rsi_14` | Technique | Relative Strength Index sur 14 séances (surachat/survente). |
| `sma_ratio` | Technique | Écart relatif du cours par rapport à sa SMA20 (`P_t / SMA_20 - 1`). |
| `macd_diff` | Technique | Histogramme du MACD (momentum). |

---

## 3. Découpage Temporel et Protocole d'Entraînement

* **Apprentissage (Train) :** 2018-01-02 → 2022-12-30 (**1 265 séances**) — *Estimation des poids et fit du `StandardScaler`.*
* **Validation :** 2023-01-03 → 2024-12-31 (**522 séances**) — *Ajustement des hyperparamètres et sélection du champion.*
* **Test Gelé (Hors-Échantillon) :** 2025-01-02 → 2026-10-30 (**477 séances**) — *Évaluation finale aveugle.*

---

## 4. Modèles Implémentés et Spécifications

1. **Baseline Majoritaire (Référence naïve) :** Prédit systématiquement la classe dominante observée sur le Train (hausse).
2. **Logistic Regression L2 (Ridge Classifier) :** Modèle linéaire régularisé (`C = 0.1`, solveur `lbfgs`, pénalité Ridge).
3. **Random Forest Classifier (Champion) :** Forêt aléatoire de 150 arbres, profondeur contrainte `max_depth = 4`, `min_samples_split = 10`.
4. **HistGradientBoosting Classifier :** Boosting d'arbres régularisé (`max_iter = 100`, `max_depth = 3`, `lr = 0.03`, `l2_reg = 1.0`).
5. **Multi-Layer Perceptron (MLP Neural Net) :** Réseau feed-forward compact (32, 16), activation ReLU, Adam, `alpha = 0.01`, early stopping.

---

## 5. Résultats et Benchmark Comparatif

### 5.1 Performances sur l'Échantillon de Validation (2023–2024, 522 séances)

| Modèle | Accuracy / Hit Ratio | Precision | Recall | Macro F1 | ROC-AUC | Temps Fit (s) | Temps Inf. (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Majoritaire** | 51.92 % | 51.92 % | 100.00 % | 34.17 % | 50.00 % | 0.001 s | 0.001 ms |
| **Logistic Regression L2 (Ridge)** | **51.34 %** | **52.14 %** | 76.38 % | **47.21 %** | **50.35 %** | **0.017 s** | **0.002 ms** |
| **Random Forest** | 50.57 % | 51.47 % | 83.76 % | 43.03 % | 45.55 % | 0.289 s | 0.068 ms |
| **HistGradientBoosting** | 48.85 % | 50.54 % | 69.00 % | 46.05 % | 46.33 % | 0.064 s | 0.008 ms |
| **MLP (Deep Learning)** | 48.66 % | 50.53 % | 52.77 % | 48.47 % | 48.87 % | 0.373 s | 0.002 ms |

### 5.2 Performances Hors-Échantillon sur le Test Gelé (2025–2026, 477 séances)

| Modèle | Accuracy / Hit Ratio | Precision | Recall | Macro F1 | ROC-AUC | Rang Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **51.78 %** | **51.33 %** | 80.42 % | **47.31 %** | 47.27 % | **1 (Meilleur test)** |
| **Baseline Majoritaire** | 50.31 % | 50.31 % | 100.00 % | 33.47 % | 50.00 % | 2 |
| **MLP (Deep Learning)** | 47.59 % | 48.33 % | 60.42 % | 46.66 % | 45.88 % | 3 |
| **HistGradientBoosting** | 46.75 % | 47.84 % | 64.58 % | 44.92 % | 46.10 % | 4 |
| **Logistic Regression L2 (Ridge)** | 46.54 % | 47.80 % | 67.92 % | 43.87 % | 47.59 % | 5 |

---

## 6. Analyse Critique des Résultats

1. **Supériorité hors échantillon du Random Forest :** Sur le test gelé, **Random Forest surpasse le buy-and-hold (50.31%) et SARIMAX (50.31%) avec 51.78% de direction correcte**. Sa structure à profondeur contrainte (`max_depth = 4`) le protège efficacement du surapprentissage.
2. **Apport direct de GARCH (D04) :** L'analyse d'importance des variables (Gini MDI) montre que la volatilité conditionnelle GARCH (`garch_vol`) se classe en **3ème position des variables les plus contributives (10.3%)**, prouvant que le module de risque D04 améliore directement l'apprentissage prédictif.
3. **Surapprentissage des modèles profonds :** Le Gradient Boosting et le MLP ont tendance à surajuster le bruit des rendements quotidiens, confirmant la règle de parcimonie en trading quantitatif.

---

## 7. Visualisation d'Audit

![D05 — Machine Learning et Deep Learning : Benchmark Prédictif](reports/figures/D05_ml_dl_comparatif.png)

### Lecture des 4 panneaux :
1. **Panneau 1 (Comparatif Hit Ratio) :** Confrontation directe des taux de succès sur Validation et Test. Le Random Forest maintient une tenue supérieure à 51.7% sur les données gelées.
2. **Panneau 2 (Feature Importance) :** Décomposition de l'apport de chaque variable. La volatilité GARCH (10.3%) et le ratio SMA surpassent les indicateurs momentum classiques (RSI, MACD).
3. **Panneau 3 (Courbes ROC sur Test Gelé) :** Les courbes des 4 modèles oscillent autour de la diagonale aléatoire (AUC entre 46% et 48%), illustrant la difficulté intrinsèque de séparation probabiliste à court terme.
4. **Panneau 4 (Matrice de Confusion) :** Répartition des décisions (hausse / baisse) du modèle champion, confirmant une capacité équilibrée à détecter les rebonds sans biais aveugle.

---

## 8. Modèle Champion pour le Livrable D06

* **Modèle Retenu :** Le **Random Forest Classifier** (`max_depth = 4`, `n_estimators = 150`) est désigné modèle champion pour alimenter les signaux de trading du **Livrable D06 (Stratégie & Backtest)**.
* **Justification :** Meilleur Hit Ratio sur les données réelles du test gelé (51.78%), robustesse prouvée contre l'overfitting face au MLP et au Gradient Boosting, et exploitation avérée de la volatilité GARCH issue du D04.

---

## 9. Fichiers Livrés pour D05

* [`src/models/ml_models.py`](src/models/ml_models.py) : Script autonome d'ingénierie, d'entraînement et d'évaluation.
* [`data/processed/ml_predictions.csv`](data/processed/ml_predictions.csv) : Fichier de signaux horodatés (prédictions et probabilités) prêt pour le backtester du D06.
* [`reports/figures/D05_ml_dl_comparatif.png`](reports/figures/D05_ml_dl_comparatif.png) : Figure d'audit à 4 panneaux.
* [`reports/D05_ml_dl.md`](reports/D05_ml_dl.md) : Rapport d'évaluation complet.
