# ⚠️ VALIDAČNÁ ANALÝZA NEURAL NETWORK MODELU

**Dátum:** 2025-10-31
**Pôvodný Sharpe:** 2.63
**Záver:** ⚠️ **MODEL JE PRAVDEPODOBNE OVERFITTED**

---

## 📊 SÚHRN VÝSLEDKOV

| Test | Očakávaný Sharpe | Skutočný Sharpe | Rozdiel |
|------|------------------|-----------------|---------|
| **Original Test** | - | **2.63** | Baseline |
| **Walk-Forward** | ~2.0 | **0.34** | **-87%** ❌ |
| **Cross-Validation** | ~2.0 | **0.39** | **-85%** ❌ |
| **Out-of-Sample** | ~2.0 | **0.59** | **-78%** ❌ |
| **Monte Carlo** | ~2.0 | **0.82** | **-69%** ❌ |
| **Permutation (random)** | ~0 | **1.69** | - ⚠️ |

---

## 🔍 DETAILNÉ VÝSLEDKY

### 1. WALK-FORWARD VALIDATION (5 folds)

| Fold | Obdobie | Test samples | Sharpe | Correlation |
|------|---------|--------------|--------|-------------|
| 1 | 2021-08 → 2022-06 | 210 | **-1.56** | -0.1146 |
| 2 | 2022-06 → 2023-04 | 210 | **0.10** | -0.0558 |
| 3 | 2023-04 → 2024-02 | 210 | **1.34** | 0.0182 |
| 4 | 2024-02 → 2024-12 | 210 | **1.33** | -0.0630 |
| 5 | 2024-12 → 2025-10 | 210 | **0.49** | 0.0540 |

**Priemer:** 0.34 ± 1.19
**Min:** -1.56
**Max:** 1.34

**Problém:** Veľká variancia! Sharpe kolíše od -1.56 po +1.34. To znamená, že model nie je stabilný.

---

### 2. TIME-SERIES CROSS-VALIDATION (5 folds)

| Fold | Train | Test | Sharpe | Correlation |
|------|-------|------|--------|-------------|
| 1 | 212 | 210 | **-0.77** | -0.0308 |
| 2 | 422 | 210 | **-0.08** | 0.1501 |
| 3 | 632 | 210 | **1.11** | 0.1110 |
| 4 | 842 | 210 | **0.93** | -0.0851 |
| 5 | 1052 | 210 | **0.74** | -0.0521 |

**Priemer:** 0.39 ± 0.79
**Stabilita:** Nízka (vysoká štandardná odchýlka)

---

### 3. OUT-OF-SAMPLE TESTING

**Training:** 2020-2023 (807 samples)

| Rok | Samples | Sharpe | Correlation | Trades |
|-----|---------|--------|-------------|--------|
| 2024 | 252 | **1.06** | -0.0437 | 49 |
| 2025 | 203 | **0.11** | -0.0677 | 43 |

**Priemer:** 0.59

---

### 4. MONTE CARLO SIMULÁCIE (100 runs)

**Štatistiky:**
- **Mean Sharpe:** 0.82 ± 0.38
- **Median:** 0.82
- **Min:** -0.20
- **Max:** 1.73
- **95% CI:** [-0.11, 1.48]

**Distribúcia:**
- **% Positive:** 95% ✅
- **% > 1.5:** 3% ❌
- **% > 2.0:** 0% ❌
- **% > 2.63:** 0% ❌

**Záver:** Ani jeden z 100 runov nedosiahol Sharpe > 2.0!

---

### 5. PERMUTATION TESTING (20 runs, random target)

| Metrika | Hodnota |
|---------|---------|
| Mean Sharpe (random) | 0.90 |
| Max Sharpe (random) | **1.69** |
| Original Sharpe | 2.63 |
| P-value | 0.0000 |

**KRITICKÝ PROBLÉM:** Random target dosiahol Sharpe 1.69!

To znamená, že aj s úplne náhodným targetom môžeme dostať slušné výsledky. Original Sharpe 2.63 je síce vyšší, ale rozdiel nie je taký výrazný ako by mal byť.

---

## 🚨 HLAVNÉ PROBLÉMY

### 1. **OVERFITTING NA TEST SETE**
- Original Sharpe 2.63
- Cross-validation priemer: 0.39
- **Rozdiel: 87%!**

→ Model sa naučil špecifiká test setu namiesto generalizovateľných vzorcov.

### 2. **NESTABILITA**
- Walk-forward Sharpe: -1.56 až +1.34
- Štandardná odchýlka: 1.19

→ Model nepredvída konzistentne. Výsledky záv isľa od obdobia.

### 3. **MALÝ TESTOVACÍ SET**
- Test: len 127 samples (~6 mesiacov)
- Train: 883 samples
- Ratio: 14%

→ Príliš malý test set umožňuje šťastné trafy.

### 4. **RANDOM BASELINE JE PRÍLIŠ DOBRÝ**
- Random target max Sharpe: 1.69
- Rozdiel oproti original: len 56%

→ Systém má príliš veľa noise, model zachytáva aj náhodné vzory.

### 5. **ŽIADNY RUN > 2.0**
- 100 Monte Carlo simulácií
- Max dosiahnutý Sharpe: 1.73
- Original: 2.63

→ Original výsledok je outlier, nie norma.

---

## 📈 POROVNANIE S XGBOOST BASELINE

| Metrika | XGBoost | Neural Net (original) | NN (validated) |
|---------|---------|----------------------|----------------|
| Test Sharpe | 1.34 | 2.63 | **0.82** |
| Cross-Val Sharpe | ? | - | **0.39** |
| Walk-Forward | ? | - | **0.34** |
| Stability | High | Low | Low |

**Záver:** XGBoost baseline (1.34) je pravdepodobne **robustnejší** než Neural Network!

---

## ✅ ČO FUNGUJE

1. **95% Monte Carlo runov je pozitívnych**
   - Model nie je úplne zlý
   - Dokáže vytvárať profit

2. **Out-of-sample 2024: Sharpe 1.06**
   - Na niektorých obdobiach funguje dobre

3. **Permutation test: Max random 1.69 << 2.63**
   - Model sa naučil niečo reálne (nie len noise)

---

## ❌ ČO NEFUNGUJE

1. **Veľká variancia výsledkov**
   - Sharpe -1.56 až +1.73
   - Nestabilný model

2. **Test set bol príliš malý**
   - 127 samples umožňuje šťastie

3. **Overfitting**
   - Train-test gap: 87%

4. **Hyperparameter tuning na test sete**
   - Custom Sharpe loss optimalizuje priamo test metriku
   - To je leaked information!

---

## 🎯 REALISTICKÝ OČAKÁVANÝ SHARPE

| Scenario | Sharpe | Pravdepodobnosť |
|----------|--------|-----------------|
| **Optimistický** | 1.0-1.5 | 10% |
| **Realistický** | 0.5-1.0 | 70% |
| **Pesimistický** | 0-0.5 | 20% |

**Best estimate:** **0.7-0.9** (Monte Carlo priemer)

---

## 💡 ODPORÚČANIA

### 1. **POUŽIŤ XGBOOST NAMIESTO NN**
- XGBoost baseline 1.34 je stabilnejší
- Jednoduchší model = menšie riziko overfittingu

### 2. **AK CHCETE POUŽIŤ NN, TAK:**

#### A) **Zväčšite test set**
- Minimálne 30% dát (nie 10%)
- Walk-forward approach

#### B) **Zmeňte loss funkciu**
- Nepoužívajte Sharpe loss (leakage!)
- MSE alebo Correlation loss

#### C) **Zjednodušte architektúru**
- Menej vrstiev (2 namiesto 3)
- Menej parametrov (50k namiesto 91k)
- Viac dropout (0.5 namiesto 0.3)

#### D) **Early stopping je tvoj priateľ**
- Zastavil sa na epoch 20 (dobre!)
- Ale validation set bol príliš malý

#### E) **Ensemble**
- Kombinovať NN + XGBoost
- Weighted average: 30% NN + 70% XGBoost

---

## 🔬 TECHNICKÉ DETAILY

### Prečo Monte Carlo < Original?

1. **Random initialization**
   - Každý run má iné váhy
   - Original mal "šťastné" váhy

2. **Early stopping variabilita**
   - Zastaví sa na rôznych epochách
   - Original sa zastavil v optimálnom bode

3. **Malý test set**
   - 127 samples = vysoká variancia
   - Niektoré runy majú smolu, iné šťastie

### Prečo Permutation baseline je vysoký?

1. **Dataset má noise**
   - Ceny majú momentum a mean reversion
   - Aj random target sa dokáže "naučiť" tieto vzory

2. **Overfitting na train set**
   - Model sa prispôsobí akémukoľvek targetu
   - 91k parametrov na 883 samples = 100 parametrov/sample!

3. **Features sú korelované**
   - 117 features, ale reálna dimensionalita je nižšia
   - Model nájde korelácie aj s random targetom

---

## 📊 FINÁLNY VERDIKT

### ⚠️ **NEURAL NETWORK MODEL JE OVERFITTED**

**Dôkazy:**
1. ❌ Cross-val Sharpe (0.39) << Test Sharpe (2.63)
2. ❌ Walk-forward Sharpe (0.34) << Test Sharpe
3. ❌ Žiadny Monte Carlo run > 2.0
4. ❌ Random target dosiahol 1.69
5. ❌ Veľká variancia (-1.56 až +1.73)

**Realistický Sharpe:** **0.7-0.9** (nie 2.63)

**Odporúčanie:**
- **Pre produkciu: XGBOOST (Sharpe 1.34)**
- Alebo: Ensemble 30% NN + 70% XGBoost

---

## 📝 LESSONS LEARNED

1. **Vždy validuj na viacerých obdobiach**
2. **Malý test set = vysoké riziko overfittingu**
3. **Custom loss functions môžu leaknúť informácie**
4. **Monte Carlo je nutnosť pre neural networks**
5. **Jednoduchší model > komplexný model**
6. **XGBoost baseline (1.34) bol v skutočnosti dobrý!**

---

**Poznámka:** Toto je brutálne úprimná analýza. Original Sharpe 2.63 bol príliš pekný na to, aby bol pravdivý. Validation to potvrdila.

**Next steps:**
1. Skúsiť ensemble approach
2. Retrain s väčším test setom (30%)
3. Použiť simpler architecture
4. Alebo jednoducho použiť XGBoost 😊
