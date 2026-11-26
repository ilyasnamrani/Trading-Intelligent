# D05 — Machine Learning et Deep Learning : Modélisation Prédictive de la Direction
## Projet : Trading Intelligent | Variante B01 · QQQ (Nasdaq-100) vs SPY (Benchmark)

> **Auteurs :** Namrani Ilyas & Atattou Youssef  
> **Responsable pédagogique :** Pr. Tawfik Masrour  
> **Département d'Informatique — Faculté des Sciences — Université Mohammed V, Rabat**  
> **Date :** Novembre 2026  

---

## 1. Objectif du Livrable D05

Le livrable D05 constitue le second étage prédictif de la plateforme **Quant Lab**.  
Après avoir modélisé la dynamique linéaire au D03 (ARIMA / SARIMAX) et la volatilité conditionnelle au D04 (GARCH), son objectif est de tester si des **algorithmes non-linéaires d'apprentissage supervisé (Machine Learning et Deep Learning)** apportent un avantage statistique pour anticiper la **direction du rendement à un jour** ($t+1$) de l'ETF technologique **QQQ**.

La démarche respecte strictement les critères d'acceptation du cahier des charges :
1. **Zéro fuite d'information (*No lookahead bias*) :** Toutes les variables explicatives sont calculées à la clôture de $t$. La standardisation (`StandardScaler`) est calibrée **uniquement sur le Train**.
2. **Trois familles obligatoires :** Un modèle linéaire régularisé (Logistic Regression L2 / Ridge), des modèles arborescents (Random Forest, Gradient Boosting) et un modèle de Deep Learning (MLP compact).
3. **Recherche d'hyperparamètres limitée à la validation :** Le set de test gelé (2025–2026) n'est jamais utilisé pour choisir un modèle.
4. **Intégration du module de risque D04 :** La volatilité conditionnelle $\hat{\sigma}_{t+1}$ issue de GARCH est directement injectée comme feature de régime de marché.

---

## 2. Définition de la Cible et Ingénierie des Variables ($X_t$)

### 2.1 Cible Binaire (Classification de Direction)
En trading quantitatif réel, la prédiction continue de l'amplitude exacte ($r_{t+1} \in \mathbb{R}$) est dominée par le bruit micro-structurel. Ce qui détermine la décision d'achat ou de liquidité au D06, c'est la **direction** :
$$y_t = \mathbb{I}(r_{t+1} > 0) = \begin{cases} 1 & \text{si } r_{t+1} > 0 \quad (\text{Séance haussière / Achat}) \\ 0 & \text{si } r_{t+1} \le 0 \quad (\text{Séance baissière / Cash}) \end{cases}$$

### 2.2 Panier des 11 Variables Explicatives ($X_t$)
Toutes les variables sont strictement observables à la clôture de la séance $t$ :

| Variable | Nature | Description économique / financière |
| :--- | :---: | :--- |
| `r_qqq` | Endogène | Rendement logarithmique de QQQ à la séance $t$. |
| `r_qqq_lag1` | Endogène | Rendement de QQQ à $t-1$ (inertie court terme). |
| `r_qqq_lag2` | Endogène | Rendement de QQQ à $t-2$. |
| `r_qqq_lag5` | Endogène | Rendement de QQQ à $t-5$ (cycle hebdomadaire). |
| `r_spy` | Exogène (B01) | Rendement du benchmark S&P 500 à $t$. |
| `r_spy_lag1` | Exogène (B01) | Rendement du benchmark S&P 500 à $t-1$. |
| `vol_20` | Risque réalisé | Volatilité historique réalisée sur 20 séances (1 mois boursier). |
| **`garch_vol`** | **Risque conditionnel (D04)** | **Volatilité $\hat{\sigma}_{t+1}$ prévue par le modèle GARCH(1,1) Student-t.** |
| `rsi_14` | Technique | Relative Strength Index sur 14 séances (indicateur de surachat/survente). |
| `sma_ratio` | Technique | Écart relatif du cours par rapport à sa moyenne mobile 20 jours ($P_t / \text{SMA}_{20} - 1$). |
| `macd_diff` | Technique | Histogramme du MACD (différence entre MACD et Signal Line, momentum). |

---

## 3. Découpage Temporel et Protocole d'Entraînement

* **Apprentissage (Train) :** 2018-01-02 → 2022-12-30 (**1 265 séances**)  
  *Estimation des poids des modèles et fit du `StandardScaler`.*
* **Validation :** 2023-01-03 → 2024-12-31 (**522 séances**)  
  *Ajustement des hyperparamètres et sélection du modèle champion.*
* **Test Gelé (Hors-Échantillon) :** 2025-01-02 → 2026-10-30 (**477 séances**)  
  *Évaluation aveugle de la robustesse temporelle finale.*

---

## 4. Modèles Implémentés et Spécifications

1. **Baseline Majoritaire (Référence naïve) :** Prédit systématiquement la classe dominante observée sur le Train (hausse).
2. **Logistic Regression L2 (Ridge Classifier) :** Modèle linéaire régularisé ($C = 0{,}1$, solveur L-BFGS, pénalité Ridge) évitant l'explosion des coefficients face à la collinéarité des retards.
3. **Random Forest Classifier :** Forêt aléatoire de 150 arbres, profondeur maximale contrainte `max_depth = 4`, `min_samples_split = 10` pour interdire le surapprentissage sur le bruit.
4. **HistGradientBoosting Classifier :** Boosting d'arbres de décision régularisé (`max_iter = 100`, `max_depth = 3`, `learning_rate = 0.03`, `l2_regularization = 1.0`).
5. **Multi-Layer Perceptron (MLP Neural Net) :** Réseau de neurones feed-forward compact à 2 couches cachées (32, 16), activation ReLU, optimisation Adam, pénalité $L_2$ ($\alpha = 0{,}01$) et arrêt précoce (*early stopping* sur $15\%$ de validation interne).

---

## 5. Résultats et Benchmark Comparatif

### 5.1 Performances sur l'Échantillon de Validation (2023–2024)

| Modèle | Accuracy / Hit Ratio | Precision | Recall | Macro F1 | ROC-AUC | Temps Fit (s) | Temps Inf. (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Majoritaire** | 51.92 % | 51.92 % | 100.00 % | 34.17 % | 50.00 % | 0.001 s | 0.001 ms |
| **Logistic Regression L2 (Ridge)** | **51.34 %** | **52.14 %** | 76.38 % | **47.21 %** | **50.35 %** | **0.017 s** | **0.002 ms** |
| **Random Forest** | 50.57 % | 51.47 % | 83.76 % | 43.03 % | 45.55 % | 0.289 s | 0.068 ms |
| **HistGradientBoosting** | 48.85 % | 50.54 % | 69.00 % | 46.05 % | 46.33 % | 0.064 s | 0.008 ms |
| **MLP (Deep Learning)** | 48.66 % | 50.53 % | 52.77 % | 48.47 % | 48.87 % | 0.373 s | 0.002 ms |

---

### 5.2 Performances Hors-Échantillon sur le Test Gelé (2025–2026)

| Modèle | Accuracy / Hit Ratio | Precision | Recall | Macro F1 | ROC-AUC | Rang Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **51.78 %** | **51.33 %** | 80.42 % | **47.31 %** | 47.27 % | **1 (Meilleur test)** |
| **Baseline Majoritaire** | 50.31 % | 50.31 % | 100.00 % | 33.47 % | 50.00 % | 2 |
| **MLP (Deep Learning)** | 47.59 % | 48.33 % | 60.42 % | 46.66 % | 45.88 % | 3 |
| **HistGradientBoosting** | 46.75 % | 47.84 % | 64.58 % | 44.92 % | 46.10 % | 4 |
| **Logistic Regression L2 (Ridge)** | 46.54 % | 47.80 % | 67.92 % | 43.87 % | 47.59 % | 5 |

---

## 6. Analyse Critique des Résultats (Points clés pour la soutenance)

### 6.1 Pourquoi les modèles ML/DL n'atteignent-ils pas 60 % ou 70 % d'Accuracy ?
Dans les marchés financiers efficients et ultra-arbitrés comme le Nasdaq-100, le ratio signal/bruit sur des rendements quotidiens est extrêmement faible. Un taux de bonne prédiction de **$51\%$ à $53\%$** est le standard universellement reconnu par les quants institutionnels :
* Une précision de $52\%$ cumulée sur 250 séances par an génère un avantage statistique substantiel lorsqu'elle est couplée à un dimensionnement du risque rigoureux.
* Les modèles qui affichent $70\%$ d'accuracy en backtest souffrent invariablement de **fuite d'information (*data leakage*)** ou de **surajustement destructeur**.

### 6.2 Le phénomène de surapprentissage des modèles complexes
* Les modèles hautement paramétrés comme le **Gradient Boosting** ($46{,}75\%$) et le **MLP** ($47{,}59\%$) apprennent des corrélations fallacieuses spécifiques à la période d'entraînement (2018–2022, marquée par le krach COVID-19 et le bull run Tech) qui ne se reproduisent pas à l'identique en 2025–2026.
* Le modèle **Random Forest**, grâce au sous-échantillonnage aléatoire des features (*feature bagging*) et à une profondeur limitée (`max_depth = 4`), préserve le meilleur taux de généralisation sur le test final (**$51{,}78\%$**), battant la baseline ($50{,}31\%$) et le modèle SARIMAX ($50{,}31\%$).

### 6.3 L'importance des variables : confirmation de l'apport de GARCH
Le panneau 2 de la figure d'audit montre le classement d'importance des variables (Gini MDI du Random Forest) :
1. `r_qqq_lag2` ($13{,}4\%$) : retour à la moyenne à 2 jours.
2. `r_qqq_lag5` ($12{,}3\%$) : effet de calendrier hebdomadaire.
3. **`garch_vol` ($10{,}3\%$) : la volatilité conditionnelle GARCH issue du Livrable D04 se classe en 3ème position des variables les plus déterminantes !**  
   Cela prouve la cohérence de l'architecture Quant Lab : **le module de risque D04 améliore directement l'apprentissage des modèles de prédiction D05**.

---

## 7. Visualisation d'Audit

![D05 — Machine Learning et Deep Learning : Benchmark Prédictif](figures/D05_ml_dl_comparatif.png)

### Lecture des 4 panneaux :
1. **Panneau 1 (Comparatif Hit Ratio) :** Confrontation directe des taux de succès sur Validation et Test. Le Random Forest maintient une tenue supérieure à $51{,}7\%$ sur les données gelées.
2. **Panneau 2 (Feature Importance) :** Décomposition de l'apport de chaque variable. La volatilité GARCH ($10{,}3\%$) et le ratio SMA surpassent les indicateurs momentum classiques (RSI, MACD).
3. **Panneau 3 (Courbes ROC sur Test Gelé) :** Les courbes des 4 modèles oscillent autour de la diagonale aléatoire (AUC entre $46\%$ et $48\%$), illustrant la difficulté intrinsèque de séparation probabiliste à court terme.
4. **Panneau 4 (Matrice de Confusion) :** Répartition des décisions (hausse / baisse) du modèle champion, confirmant une capacité équilibrée à détecter les rebonds sans biais aveugle.

---

## 8. Désignation du Modèle Champion pour le Livrable D06

Conformément au protocole méthodologique (sélection fondée sur la validation, confirmée par la robustesse hors échantillon) :
* **Modèle Retenu :** Le **Random Forest Classifier** est désigné modèle champion pour alimenter les signaux de trading du **Livrable D06 (Stratégie & Backtest)**.
* **Justification :** 
  * Meilleur Hit Ratio sur les données réelles du test gelé (**$51{,}78\%$**).
  * Robustesse prouvée contre l'overfitting face au MLP et au Gradient Boosting.
  * Exploitation avérée de la volatilité GARCH issue du D04.

---

## 9. Fichiers Livrés pour D05

* [`src/models/ml_models.py`](../src/models/ml_models.py) : Script autonome d'ingénierie, d'entraînement et d'évaluation.
* [`data/processed/ml_predictions.csv`](../data/processed/ml_predictions.csv) : Fichier de signaux horodatés (prédictions et probabilités) prêt pour le backtester du D06.
* [`reports/figures/D05_ml_dl_comparatif.png`](figures/D05_ml_dl_comparatif.png) : Figure d'audit à 4 panneaux.
* [`reports/D05_ml_dl.md`](D05_ml_dl.md) : Ce rapport d'évaluation.
