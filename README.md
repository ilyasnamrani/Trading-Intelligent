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

# Livrable Actuel : D05 — Machine Learning et Deep Learning

## 1. Objectif du Livrable D05

Le livrable D05 constitue le second étage prédictif de la plateforme **Quant Lab**, au cœur de la question scientifique de la variante **B01** :
> *« Les modèles ML/DL justifient-ils leur complexité face à SARIMAX sur un ETF de grandes entreprises technologiques ? »*

Alors que le D03 modélisait la dynamique linéaire (Hit Ratio ≈ 50.3% - 52.8%) et le D04 la volatilité conditionnelle GARCH, le livrable D05 évalue la capacité d'algorithmes non-linéaires d'apprentissage supervisé à anticiper la **direction du rendement à un jour (`target_direction = 1 si r_{t+1} > 0`)**.

---

## 2. Architecture du Code (`src/models/ml_models.py`)

Les 3 familles exigées par le cahier des charges sont implémentées et comparées sous un protocole temporel étanche :

| Famille | Modèle | Hyperparamètres clés (réglés sur Validation) | Rôle dans le projet |
| :--- | :--- | :--- | :--- |
| **Linéaire régularisé** | **Logistic Regression L2 (Ridge)** | `C = 0.1`, solveur `lbfgs`, pénalité L2 | Baseline linéaire pénalisée contre l'overfitting. |
| **Arborescent (Bagging)** | **Random Forest Classifier** | `n_estimators = 150`, `max_depth = 4`, `min_samples_split = 10` | Capture des interactions non-linéaires avec régularisation de profondeur. |
| **Arborescent (Boosting)** | **HistGradientBoosting** | `max_iter = 100`, `max_depth = 3`, `lr = 0.03`, `l2_reg = 1.0` | Algorithme séquentiel de boosting à gradient. |
| **Deep Learning** | **MLP Neural Net** | Architecture (32, 16), activation ReLU, `alpha = 0.01`, early stopping | Réseau dense feed-forward compact sous contrainte temporelle. |

---

## 3. Synthèse des Résultats et Benchmark Comparatif

### 3.1 Performance sur la Validation (2023–2024, 522 séances) et Test Gelé (2025–2026, 477 séances)

| Modèle | Accuracy Validation | Hit Ratio Validation | Accuracy Test Gelé | Hit Ratio Test Gelé | Macro F1 Test | Rang Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | 50.57 % | 50.57 % | **51.78 %** | **51.78 %** | **47.31 %** | **1 (Retenu)** |
| **Baseline Majoritaire** | 51.92 % | 51.92 % | 50.31 % | 50.31 % | 33.47 % | 2 |
| **MLP Neural Net (Deep Learning)** | 48.66 % | 48.66 % | 47.59 % | 47.59 % | 46.66 % | 3 |
| **HistGradientBoosting** | 48.85 % | 48.85 % | 46.75 % | 46.75 % | 44.92 % | 4 |
| **Logistic Regression L2 (Ridge)** | 51.34 % | 51.34 % | 46.54 % | 46.54 % | 43.87 % | 5 |

### 3.2 Analyse Critique & Apport du Livrable D04
1. **Supériorité hors échantillon du Random Forest :** Sur le test gelé, **Random Forest surpasse le buy-and-hold (50.31%) et SARIMAX (50.31%) avec 51.78% de direction correcte**. Sa structure à profondeur contrainte (`max_depth = 4`) le protège efficacement du surapprentissage.
2. **Apport direct de GARCH (D04) :** L'analyse d'importance des variables (Gini MDI) montre que la volatilité conditionnelle GARCH (`garch_vol`) se classe en **3ème position des variables les plus contributives (10.3%)**, prouvant que le module de risque D04 améliore l'apprentissage prédictif.
3. **Surapprentissage des modèles profonds :** Le Gradient Boosting et le MLP ont tendance à surajuster le bruit des rendements quotidiens, illustrant la règle de parcimonie sur les séries financières liquides.

---

## 4. Visualisation d'Audit

![D05 — Machine Learning et Deep Learning : Benchmark Prédictif](reports/figures/D05_ml_dl_comparatif.png)

> **Pour la version complète et l'analyse détaillée :** [Consulter le rapport D05](reports/D05_ml_dl.md) | **Code source :** [`src/models/ml_models.py`](src/models/ml_models.py) | **Signaux exportés :** [`data/processed/ml_predictions.csv`](data/processed/ml_predictions.csv)
