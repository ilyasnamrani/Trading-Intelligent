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

> **Question centrale :** *« Une Intelligence Artificielle (Machine Learning / Deep Learning) peut-elle prédire si le Nasdaq-100 (QQQ) monte ou descend demain, mieux qu'un simple pile ou face et mieux que le modèle SARIMAX ? »*

* **Variable cible binaire (`target_direction`) :**
  * `1` si le rendement de demain est positif ($r_{t+1} > 0 \rightarrow$ Signal d'Achat)
  * `0` si le rendement de demain est négatif ou nul ($r_{t+1} \le 0 \rightarrow$ Signal de Cash)

---

## 2. Respect Strict des Critères d'Acceptation

Conformément aux exigences du cahier des charges (Pr. Tawfik Masrour) :

1. **Baselines Communes :**  
   Comparaison systématique à un point de repère zéro : la **Baseline Majoritaire** (Buy-and-Hold passif, parier systématiquement la hausse) et le modèle économétrique **SARIMAX(1,0,0) du D03**.
2. **Recherche d'Hyperparamètres Limitée à la Validation (Zéro Triche) :**  
   * **Train (2018–2022, 1 265 séances) :** Entraînement des poids et calibration du scaler (`StandardScaler`).
   * **Validation (2023–2024, 522 séances) :** Réglage exclusif des hyperparamètres (profondeur des arbres, neurones, pénalités L2).
   * **Test Gelé (2025–2026, 477 séances) :** Évaluation finale aveugle hors-échantillon sans aucun réajustement.

---

## 3. Les 3 Familles de Modèles en Compétition

| Famille | Modèle | Principe simple |
| :--- | :--- | :--- |
| **Linéaire régularisé** | **Logistic Regression L2 (Ridge)** | Modèle linéaire avec un « frein » mathématique (L2) pour éviter le surapprentissage. |
| **Arborescent (Champion)** | **Random Forest Classifier** | Comité de 150 arbres de décision qui votent ensemble (`max_depth = 4`). |
| **Arborescent (Boosting)** | **HistGradientBoosting** | Arbres séquentiels apprenant pas à pas de leurs erreurs passées. |
| **Deep Learning** | **MLP Neural Net** | Réseau de neurones artificiels compact (2 couches : 32 et 16 neurones, Dropout). |

---

## 4. Résultats Comparatifs : L'Examen Final sur le Test Gelé (2025–2026)

| Modèle | Famille | Hit Ratio Test | Macro F1 | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | **Arborescent** | **51.78 %** | **47.31 %** | 🏆 **CHAMPION : Bat la baseline et bat SARIMAX.** |
| **Baseline Majoritaire** | Référence naïve | 50.31 % | 33.47 % | Point de repère passif (Buy & Hold). |
| **SARIMAX (D03)** | Économétrie | 50.31 % | — | Égalité avec la baseline sur le test. |
| **MLP Neural Net** | Deep Learning | 47.59 % | 46.66 % | Échoue : trop complexe, surajuste le bruit. |
| **HistGradientBoosting** | Boosting | 46.75 % | 44.92 % | Échoue : surapprentissage sur le passé. |
| **Logistic Regression L2** | Linéaire | 46.54 % | 43.87 % | Échoue : trop rigide face aux non-linéarités. |

---

## 5. Ce qu'il faut retenir pour la Soutenance

1. **Random Forest est le modèle Champion :** Avec **51.78 % de bonne direction** sur le test aveugle, il résiste au bruit boursier et offre un avantage prédictif réel face aux 50.31% de la baseline.
2. **L'apport direct du Livrable D04 (GARCH) :** L'analyse d'importance des variables montre que **la volatilité conditionnelle GARCH (`garch_vol`) est la 3ème variable la plus utile (10.3%)** pour aider le Random Forest à décider.
3. **Le piège du surapprentissage :** Les modèles complexes (Deep Learning / Boosting) apprennent le bruit passé et échouent sur le futur, prouvant la supériorité de la parcimonie en trading quantitatif.

---

## 6. Visualisation d'Audit

![D05 — Machine Learning et Deep Learning : Benchmark Prédictif](reports/figures/D05_ml_dl_comparatif.png)

> **Rapport complet détaillé :** [Consulter le rapport D05](reports/D05_ml_dl.md) | **Code source :** [`src/models/ml_models.py`](src/models/ml_models.py) | **Signaux exportés pour D06 :** [`data/processed/ml_predictions.csv`](data/processed/ml_predictions.csv)
