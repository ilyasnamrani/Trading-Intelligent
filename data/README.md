# Dictionnaire et Schéma des Données de Marché

> **Projet :** Trading Intelligent — Variante B01 (QQQ / Nasdaq-100)  
> **Source primaire :** Yahoo Finance (via API `yfinance`)  
> **Mode de secours :** Simulation brownienne géométrique (`src/generate_snapshots.py`)

---

## 1. Provenance et Conventions

* **Actifs collectés :**
  * `QQQ` : Invesco QQQ Trust (Grandes capitalisations technologiques du Nasdaq-100).
  * `SPY` : SPDR S&P 500 ETF Trust (Benchmark actions américain large).
* **Fréquence :** Données journalières (clôture US, 16h00 EST).
* **Plage temporelle :** 2 janvier 2018 au 30 octobre 2026 (2 304 séances).
* **Convention d'ajustement :** Prix ajustés automatiquement (`auto_adjust=True`) pour neutraliser les dividendes détachés et les divisions d'actions (*splits*).

---

## 2. Données Brutes (`data/raw/`)

Chaque fichier (`QQQ_raw.csv` et `SPY_raw.csv`) contient la structure standardisée OHLCV :

| Colonne | Type | Description |
| :--- | :---: | :--- |
| `Date` | Datetime (Index) | Date de la séance (format `YYYY-MM-DD`, jours ouvrés) |
| `Open` | Flottant (USD) | Cours d'ouverture de la séance |
| `High` | Flottant (USD) | Plus haut cours atteint durant la séance |
| `Low` | Flottant (USD) | Plus bas cours atteint durant la séance |
| `Close` | Flottant (USD) | Cours de clôture officiel ajusté |
| `Volume` | Entier | Volume total de titres échangés |

---

## 3. Dataset Enrichi des Variables (`data/processed/features.csv`)

Ce fichier contient **2 283 séances × 9 colonnes** prêtes pour l'entraînement des modèles économétriques et d'apprentissage automatique.

| Colonne | Type | Formule mathématique | Description & Usage |
| :--- | :---: | :--- | :--- |
| `Date` | Index | `YYYY-MM-DD` | Date de référence de l'observation à l'instant `t` |
| `close_qqq` | Niveau | `P_QQQ[t]` | Prix de clôture ajusté de QQQ |
| `close_spy` | Niveau | `P_SPY[t]` | Prix de clôture ajusté de SPY (Benchmark) |
| `r_qqq` | Variable | `ln( P_QQQ[t] / P_QQQ[t-1] )` | Rendement logarithmique journalier de QQQ |
| `r_spy` | Variable | `ln( P_SPY[t] / P_SPY[t-1] )` | Rendement logarithmique journalier de SPY |
| `r_qqq_lag1` | Entrée | `r_qqq[t-1]` | Rendement de QQQ retardé de 1 jour (momentum court terme) |
| `r_qqq_lag5` | Entrée | `r_qqq[t-5]` | Rendement de QQQ retardé de 5 jours (effet hebdomadaire) |
| `r_spy_lag1` | Entrée | `r_spy[t-1]` | **Variable exogène retardée** pour SARIMAX (B01) |
| `vol_20` | Entrée | Écart-type glissant 20j de `r_qqq` | Volatilité réalisée historique (20 séances) |
| `target` | **Cible** | `r_qqq[t+1]` | **Rendement de la séance suivante** (évaluation uniquement) |

---

## 4. Règles de Non-Fuite d'Information (*No Lookahead*)

* Toutes les variables en entrée (`r_qqq_lag1`, `r_qqq_lag5`, `r_spy_lag1`, `vol_20`) sont calculées **strictement à partir d'informations connues à la clôture de la séance `t`**.
* La variable `target` est décalée d'un cran dans le futur (`shift(-1)`) et n'est **jamais injectée comme prédicteur** lors de l'entraînement des modèles.
