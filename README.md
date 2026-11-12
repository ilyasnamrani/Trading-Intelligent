# QUANT LAB — Trading Intelligent (B01 · QQQ)

> **Projet :** Trading Intelligent — De la donnée de marché à une application de trading quantitatif  
> **Responsable pédagogique :** Pr. Tawfik Masrour | Département d'informatique — FS Rabat  
> **Actif étudié :** **QQQ** (Invesco QQQ Trust / Nasdaq-100) | **Benchmark :** **SPY** (S&P 500)  
> **Binôme :** Amrani Ilyas & Atattou Youssef  

---

### 📌 Documents de Référence & Navigation des Livrables

* 📄 **Document Fondateur :** [Consulter le Cahier des Charges Intégral du Projet](reports/cahier_des_charges.md)

| Livrable | Statut | Accès au document complet |
| :--- | :---: | :--- |
| **D01 — Cadrage & Architecture** | ✅ Validé | [Consulter le rapport D01](reports/D01_cadrage.md) |
| **D02 — Données, Ingestion & Audit** | 🚀 **Actuel (ci-dessous)** | *Affiché sur cette page* |
| **D03 — Économétrie de la moyenne** | ⏳ À venir | `reports/D03_econometrie.md` |
| **D04 — Module de Risque (GARCH)** | ⏳ À venir | `reports/D04_risque.md` |
| **D05 — Machine Learning & Deep Learning** | ⏳ À venir | `reports/D05_ml_dl.md` |
| **D06 — Stratégie & Backtesting** | ⏳ À venir | `reports/D06_backtest.md` |
| **D07 — Application Web (Streamlit)** | ⏳ À venir | Code `app.py` |
| **D08 — Paper Trading (5 séances)** | ⏳ À venir | `reports/D08_paper_trading.md` |
| **D09 — Reproductibilité & Tests** | ⏳ À venir | `tests/` & `pytest` |
| **D10 — Rapport Final & Soutenance** | ⏳ À venir | `reports/final_report.pdf` |

---

# Livrable Actuel : D02 — Données, Ingestion et Audit Qualité

## 1. Objectif du Livrable D02

Le livrable D02 constitue le socle fondamental de toute la chaîne quantitative du projet.
Son objectif est triple :

1. **Collecter et Sécuriser :** Télécharger l'historique complet de **QQQ** et **SPY** (2018–2026) et créer un snapshot local garantissant l'exécution 100% hors-ligne (mode démo imposé).
2. **Auditer rigoureusement :** Vérifier l'intégrité mathématique et temporelle des séries (zéro NaN, zéro doublon, cohérence stricte des barres OHLC, volume positif).
3. **Préparer les variables (*Feature Engineering*) :** Transformer les cours bruts en rendements logarithmiques, variables retardées et métriques de volatilité sans jamais introduire de biais d'anticipation (*no lookahead bias*).

---

## 2. Relation directe avec l'objectif du projet

La problématique scientifique de la variante **B01** est :
> *« Les modèles anticipant le risque améliorent-ils les décisions sur un ETF technologique volatil (QQQ) ? »*

Pour mesurer si la gestion du risque protège le capital lors des crises, **tous les modèles futurs dépendent directement de ce livrable D02** :
* **D03 (ARIMA / SARIMAX) :** Nécessite les rendements stationnaires `r_qqq` et la variable exogène retardée du marché `r_spy_lag1`.
* **D04 (ARCH / GARCH) :** Modélise la variance conditionnelle directement sur les résidus des rendements `r_qqq`.
* **D05 (Machine Learning & MLP) :** S'entraîne sur le vecteur de variables calculées (`r_qqq_lag1`, `r_qqq_lag5`, `vol_20`).
* **D06 (Backtesting) :** Évalue la performance nette face au benchmark passif `SPY`.

---

## 3. Données et règles de traçabilité

### 3.1 Actifs sélectionnés
* **Actif cible :** `QQQ` (*Invesco QQQ Trust*) — panier des 100 leaders technologiques et de croissance américains (Apple, Microsoft, NVIDIA, Amazon, Alphabet, Meta...).
* **Benchmark commun :** `SPY` (*SPDR S&P 500 ETF Trust*) — baromètre large du marché actions américain.

### 3.2 Découpage temporel imposé
* **Apprentissage (Train) :** 2018 – 2022
* **Validation (Tuning des hyperparamètres) :** 2023 – 2024
* **Test hors-échantillon (Test final gelé) :** 1er janvier 2025 – 30 octobre 2026
* **Volume total :** **2 304 séances de bourse** réelles/ouvrables.

### 3.3 Pourquoi utiliser les prix ajustés (`Adj Close`) ?
Le prix de clôture brut (`Close`) est trompeur :
* **Splits d'actions :** Une division par deux du nominal divise le prix par 2 sans aucune perte pour l'investisseur.
* **Dividendes :** Le versement d'un dividende réduit mécaniquement le cours de l'action.
Le prix ajusté (`Adj Close`) corrige ces événements pour que chaque variation reflète un gain ou une perte économique réel.

---

## 4. Résultats de l'Audit Qualité

L'audit automatique exécuté par `src/data.py` valide les contrôles suivants :

| Contrôle Qualité | QQQ (Actif Cible) | SPY (Benchmark) | Statut de Conformité |
| :--- | :---: | :---: | :---: |
| **Plage temporelle** | 2018-01-02 → 2026-10-30 | 2018-01-02 → 2026-10-30 | ✅ Conforme (2 304 séances) |
| **Valeurs manquantes (NaN)** | 0 | 0 | ✅ Aucune donnée manquante |
| **Doublons d'horodatage** | 0 | 0 | ✅ Index temporel strictement unique |
| **Cohérence OHLC (`Low <= Close <= High`)** | 0 anomalie | Conforme (données réelles) | ✅ Barres de prix valides |
| **Volumes nuls** | 0 | 0 | ✅ Liquidité continue constatée |
| **Rendements journaliers extrêmes (> 15%)** | 0 | 0 | ✅ Aucune aberration ponctuelle |
| **Statut d'acceptation global** | **PROPRE & VALIDÉ** | **PROPRE & VALIDÉ** | ✅ **Prêt pour modélisation** |

---

## 5. Ingénierie des Variables (*Feature Engineering*)

Afin d'éviter tout biais d'anticipation (*lookahead bias*), toute information connue à l'instant `t` est strictement calculée sur les séances antérieures ou égales à `t`.

```text
# Variable cible à prédire (strictement décalée au jour suivant t+1) :
target_t = ln( P_QQQ[t+1] / P_QQQ[t] )
```

### Dictionnaire du Dataset produit (`data/processed/features.csv`) :

| Variable | Type | Formule de calcul | Rôle dans l'application |
| :--- | :---: | :--- | :--- |
| `close_qqq` | Niveau | Cours ajusté QQQ | Base de calcul des rendements et du capital |
| `close_spy` | Niveau | Cours ajusté SPY | Référence de valorisation du benchmark |
| `r_qqq` | Entrée | `ln( P_QQQ[t] / P_QQQ[t-1] )` | Rendement continu journalier de QQQ |
| `r_spy` | Entrée | `ln( P_SPY[t] / P_SPY[t-1] )` | Rendement continu journalier du S&P 500 |
| `r_qqq_lag1` | Entrée | `r_qqq[t-1]` | Signal d'inertie / momentum à 1 jour |
| `r_qqq_lag5` | Entrée | `r_qqq[t-5]` | Effet d'autocorrélation hebdomadaire (5 séances) |
| `r_spy_lag1` | Entrée | `r_spy[t-1]` | **Variable exogène retardée imposée pour SARIMAX** |
| `vol_20` | Entrée | Écart-type glissant de `r_qqq` (20j) | Indicateur de régime de volatilité récente |
| `target` | **Cible** | `r_qqq[t+1]` | **Variable à prédire (jamais injectée en entrée)** |

**Dataset final exporté :** **2 283 séances exploitables × 9 colonnes** (après suppression des lignes d'initialisation de la fenêtre de 20 jours).

---

## 6. Visualisations et Audit Graphique

Le graphique d'audit à 4 panneaux est généré automatiquement par le pipeline :

![D02 — Audit Graphique QQQ vs SPY](reports/figures/D02_audit_donnees.png)

### Analyse des 4 panneaux :
1. **Prix normalisés (Base 100 en 2018) :** Compare la trajectoire de croissance de QQQ par rapport au benchmark SPY. Permet d'observer les phases de surperformance technologique et les corrections de marché.
2. **Rendements journaliers de QQQ :** Met en évidence les alternances entre régimes de calme plat et régimes de chocs brutaux.
3. **Volatilité réalisée 20 jours (annualisée) :** Illustre la propriété de *clustering de volatilité* (les périodes agitées succèdent aux périodes agitées), ce qui justifie mathématiquement le recours à un modèle **GARCH** au livrable D04.
4. **Distribution des rendements vs Loi normale :** Révèle des queues épaisses (*fat tails*, leptokurticité). Les chocs extrêmes sont plus fréquents que dans une loi gaussienne, justifiant le calcul de la **Value-at-Risk (VaR)** et de l'**Expected Shortfall (ES)**.

---

## 7. Reproductibilité et Mode Démonstration

L'ensemble de ce livrable est entièrement reproductible avec les commandes suivantes :

```bash
# 1. Installation des dépendances requises
pip install yfinance pandas numpy matplotlib seaborn

# 2. Exécution du pipeline complet (téléchargement, audit, features, graphiques)
python src/data.py

# 3. En cas d'indisponibilité réseau (Mode démonstration hors-ligne)
python src/generate_snapshots.py
python src/data.py
```

*Les données brutes auditées sont archivées localement dans `data/raw/` et le jeu de variables final dans `data/processed/features.csv`.*
